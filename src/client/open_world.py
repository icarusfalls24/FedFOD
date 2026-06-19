"""
FedFOD Open-World Detection Module
===================================

CLIP-based open-world FOD detection with curated text prompts for 15 FOD
classes, embedding-based classification, and novelty detection against a
prototype bank.
"""

import io
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import clip as openai_clip
except ImportError:
    openai_clip = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

logger = logging.getLogger(__name__)

# FOD class name mapping (shared with inference module)
FOD_CLASS_NAMES: Dict[int, str] = {
    0: "metal_fastener",
    1: "wire_fragment",
    2: "rubber_gasket",
    3: "plastic_debris",
    4: "tool_wrench",
    5: "tool_screwdriver",
    6: "pavement_chunk",
    7: "luggage_fragment",
    8: "engine_blade",
    9: "safety_cone",
    10: "bird_remains",
    11: "ice_chunk",
    12: "tire_fragment",
    13: "light_cover",
    14: "unknown_fod",
}


# ---------------------------------------------------------------------------
# PrototypePayload
# ---------------------------------------------------------------------------
@dataclass
class PrototypePayload:
    """Compact novelty packet ≤ 10 KB for server transmission."""

    embedding: np.ndarray  # 512-d CLIP embedding
    detection_info: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    payload_size_bytes: int = 0

    def __post_init__(self):
        self.payload_size_bytes = self._estimate_size()

    def _estimate_size(self) -> int:
        embed_bytes = self.embedding.nbytes
        info_bytes = len(json.dumps(self.detection_info, default=str).encode())
        meta_bytes = len(json.dumps(self.metadata, default=str).encode())
        return embed_bytes + info_bytes + meta_bytes + 8  # 8 for timestamp

    def to_bytes(self) -> bytes:
        """Serialise to compact bytes representation."""
        buf = io.BytesIO()
        np.save(buf, self.embedding.astype(np.float16))  # half-precision
        embed_bytes = buf.getvalue()
        payload = {
            "e": embed_bytes.hex(),
            "d": self.detection_info,
            "m": self.metadata,
            "t": self.timestamp,
        }
        return json.dumps(payload, default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# FODTextPrompts
# ---------------------------------------------------------------------------
class FODTextPrompts:
    """Curated CLIP text templates for all 15 FOD classes + novelty prompts."""

    def __init__(self):
        self.KNOWN_PROMPTS: Dict[int, List[str]] = {
            0: [
                "a metal fastener on a runway",
                "a titanium bolt on airport tarmac",
                "small metallic bolt on concrete runway",
                "a steel screw or nut on the runway surface",
                "shiny metal rivet lying on airport pavement",
            ],
            1: [
                "a wire fragment on the runway",
                "a piece of metal wire on airport tarmac",
                "thin wire debris on concrete runway surface",
                "a broken wire strand on the runway",
            ],
            2: [
                "a rubber gasket on the runway",
                "a black rubber seal on airport tarmac",
                "an O-ring gasket lying on runway pavement",
                "a circular rubber component on the runway",
            ],
            3: [
                "plastic debris on the runway",
                "a piece of broken plastic on airport tarmac",
                "white plastic fragment on concrete runway",
                "shattered plastic cover on the runway surface",
            ],
            4: [
                "a wrench tool on the runway",
                "a metal wrench lying on airport tarmac",
                "an adjustable spanner on concrete runway surface",
                "a mechanics wrench left on the runway",
                "a chrome wrench on airport pavement",
            ],
            5: [
                "a screwdriver on the runway",
                "a flathead screwdriver on airport tarmac",
                "a Phillips screwdriver on concrete runway",
                "a hand tool screwdriver on the runway surface",
            ],
            6: [
                "a pavement chunk on the runway",
                "broken concrete piece on airport tarmac",
                "a fragment of asphalt on the runway surface",
                "crumbled pavement debris on the runway",
            ],
            7: [
                "a luggage fragment on the runway",
                "a broken suitcase piece on airport tarmac",
                "luggage debris on the runway surface",
                "a torn fabric piece from baggage on the runway",
            ],
            8: [
                "an engine blade on the runway",
                "a turbine blade fragment on airport tarmac",
                "a jet engine fan blade on concrete runway",
                "a metal engine component on the runway surface",
                "a curved turbine blade lying on airport pavement",
            ],
            9: [
                "a safety cone on the runway",
                "an orange traffic cone on airport tarmac",
                "a pylons marker on the runway surface",
                "a bright orange cone on the runway",
            ],
            10: [
                "bird remains on the runway",
                "a dead bird on airport tarmac",
                "bird strike debris on concrete runway",
                "avian remains on the runway surface",
            ],
            11: [
                "an ice chunk on the runway",
                "a piece of ice on airport tarmac",
                "frozen debris on concrete runway surface",
                "an icy formation on the runway",
            ],
            12: [
                "a tire fragment on the runway",
                "rubber tire debris on airport tarmac",
                "a piece of blown tire on concrete runway",
                "shredded tire rubber on the runway surface",
                "a black tire tread on airport pavement",
            ],
            13: [
                "a light cover on the runway",
                "a broken runway light lens on airport tarmac",
                "a plastic light fixture cover on the runway",
                "a glass light cover on concrete runway surface",
            ],
            14: [
                "unknown foreign object debris on the runway",
                "unidentified debris on airport tarmac",
                "an unknown object on concrete runway surface",
                "unrecognized foreign object on the runway",
            ],
        }

        self.NOVEL_PROMPTS: List[str] = [
            "foreign object on runway",
            "debris on airport tarmac",
            "unknown object on runway surface",
            "unidentified foreign object debris on airport pavement",
            "anomalous object on the runway",
        ]

    def get_prompts(self, class_id: int) -> List[str]:
        """Return text prompts for a specific class."""
        return self.KNOWN_PROMPTS.get(class_id, self.NOVEL_PROMPTS)

    def get_all_prompts(self) -> Dict[int, List[str]]:
        """Return the full prompt dictionary."""
        return dict(self.KNOWN_PROMPTS)


# ---------------------------------------------------------------------------
# CLIPOpenWorldDetector
# ---------------------------------------------------------------------------
class CLIPOpenWorldDetector:
    """CLIP-based open-world FOD detection and novelty identification."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        cosine_threshold: float = 0.70,
    ):
        if openai_clip is None:
            raise ImportError(
                "OpenAI CLIP is required – install with: pip install git+https://github.com/openai/CLIP.git"
            )
        if PILImage is None:
            raise ImportError("Pillow is required – install with: pip install Pillow")

        self.clip_model_name = clip_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cosine_threshold = cosine_threshold

        # Load CLIP
        self.clip_model, self.clip_preprocess = openai_clip.load(
            clip_model_name, device=self.device
        )
        self.clip_model.eval()

        # Build text prompt bank
        self.fod_prompts = FODTextPrompts()
        self.class_names = FOD_CLASS_NAMES

        # Precompute text embeddings
        self._text_embeddings: Dict[int, torch.Tensor] = {}
        self._novel_text_embedding: Optional[torch.Tensor] = None
        self._precompute_text_embeddings()

        logger.info(
            "CLIPOpenWorldDetector initialised – model=%s device=%s threshold=%.2f",
            clip_model_name,
            self.device,
            cosine_threshold,
        )

    # -- public API ----------------------------------------------------------

    def compute_fod_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Encode an image crop into a 512-d CLIP embedding (L2 normalised)."""
        image_tensor = self._preprocess_image(image_crop)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def classify_known(
        self, embedding: np.ndarray
    ) -> Tuple[int, str, float]:
        """Classify an embedding against all known FOD class text embeddings.

        Returns:
            (best_class_id, class_name, cosine_similarity)
        """
        query = torch.from_numpy(embedding).float().to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        best_class_id = -1
        best_class_name = "unknown_fod"
        best_similarity = -1.0

        for class_id, text_emb in self._text_embeddings.items():
            # text_emb: (num_prompts, 512)
            similarities = torch.mm(query, text_emb.T).squeeze(0)  # (num_prompts,)
            max_sim = similarities.max().item()
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_class_id = class_id
                best_class_name = self.class_names.get(class_id, f"class_{class_id}")

        return best_class_id, best_class_name, float(best_similarity)

    def detect_novelty(
        self,
        embedding: np.ndarray,
        prototype_bank: Dict[str, np.ndarray],
    ) -> Tuple[bool, str, float]:
        """Determine whether the embedding represents a novel (unseen) class.

        Args:
            embedding: 512-d query embedding.
            prototype_bank: ``{class_name: prototype_embedding}`` dict.

        Returns:
            (is_novel, nearest_class_name, distance)
            ``is_novel`` is True when max cosine similarity < cosine_threshold.
        """
        if not prototype_bank:
            return True, "none", 0.0

        query = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_class = "none"
        best_similarity = -1.0

        for class_name, proto in prototype_bank.items():
            proto_norm = proto / (np.linalg.norm(proto) + 1e-8)
            sim = float(np.dot(query, proto_norm))
            if sim > best_similarity:
                best_similarity = sim
                best_class = class_name

        is_novel = best_similarity < self.cosine_threshold
        distance = 1.0 - best_similarity  # cosine distance
        return is_novel, best_class, float(distance)

    def generate_novelty_packet(
        self,
        embedding: np.ndarray,
        detection: Any,
        metadata: Dict,
    ) -> PrototypePayload:
        """Create a compact PrototypePayload (≤ 10 KB) for server transmission."""
        # Extract detection info safely
        detection_info: Dict[str, Any] = {}
        if hasattr(detection, "bbox"):
            detection_info["bbox"] = (
                detection.bbox.tolist()
                if isinstance(detection.bbox, np.ndarray)
                else list(detection.bbox)
            )
        if hasattr(detection, "class_id"):
            detection_info["class_id"] = int(detection.class_id)
        if hasattr(detection, "class_name"):
            detection_info["class_name"] = str(detection.class_name)
        if hasattr(detection, "confidence"):
            detection_info["confidence"] = float(detection.confidence)

        # Quantise embedding to float16 for size savings
        embedding_f16 = embedding.astype(np.float16)

        payload = PrototypePayload(
            embedding=embedding_f16,
            detection_info=detection_info,
            metadata=metadata,
            timestamp=time.time(),
        )

        # Verify size constraint (≤ 10 KB)
        if payload.payload_size_bytes > 10_240:
            # Truncate metadata to fit
            truncated_meta = {"source": metadata.get("source", "unknown")}
            payload = PrototypePayload(
                embedding=embedding_f16,
                detection_info=detection_info,
                metadata=truncated_meta,
                timestamp=time.time(),
            )
            logger.warning(
                "Novelty packet exceeded 10KB – truncated metadata to %d bytes",
                payload.payload_size_bytes,
            )

        return payload

    # -- private helpers -----------------------------------------------------

    def _precompute_text_embeddings(self) -> None:
        """Encode all FOD text prompts into L2-normalised 512-d embeddings."""
        all_prompts = self.fod_prompts.get_all_prompts()

        for class_id, prompts in all_prompts.items():
            tokens = openai_clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
            self._text_embeddings[class_id] = text_features.float()

        # Also encode novel prompts
        novel_tokens = openai_clip.tokenize(
            self.fod_prompts.NOVEL_PROMPTS
        ).to(self.device)
        with torch.no_grad():
            novel_features = self.clip_model.encode_text(novel_tokens)
        novel_features = novel_features / novel_features.norm(
            dim=-1, keepdim=True
        )
        self._novel_text_embedding = novel_features.float()

        logger.info(
            "Precomputed text embeddings for %d classes + %d novel prompts",
            len(all_prompts),
            len(self.fod_prompts.NOVEL_PROMPTS),
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert an OpenCV BGR numpy image to a CLIP-ready tensor."""
        # Convert BGR → RGB
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image.copy()

        # Convert to PIL Image for CLIP preprocessing
        pil_img = PILImage.fromarray(rgb.astype(np.uint8))
        tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        return tensor
