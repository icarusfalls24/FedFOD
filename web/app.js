/* ==========================================================================
   FedFOD Platform JS Core — REST APIs, WebSockets & Visualization
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Current application state
    const appState = {
        activeTab: "dashboard",
        trainingActive: false,
        roundsChart: null,
        wsLogs: null,
        wsMetrics: null,
        selectedImageFile: null,
        currentConfigType: "global", // 'global', 'airport_A', 'airport_B'
        apiBase: `${window.location.protocol}//${window.location.host}`,
        wsBase: `ws://${window.location.host}`,
    };

    // DOM Elements
    const elements = {
        navItems: document.querySelectorAll(".nav-item"),
        tabPanels: document.querySelectorAll(".tab-panel"),
        pageTitle: document.getElementById("page-title"),
        pageSubtitle: document.getElementById("page-subtitle"),
        
        // Header
        trainingStatusBadge: document.getElementById("training-status-badge"),
        
        // Dashboard Stats
        activeClientsVal: document.getElementById("active-clients-val"),
        currentRoundVal: document.getElementById("current-round-val"),
        totalRoundsVal: document.getElementById("total-rounds-val"),
        giniVal: document.getElementById("gini-coefficient-val"),
        uptimeVal: document.getElementById("system-uptime-val"),
        logConsole: document.getElementById("log-console-area"),
        clearLogsBtn: document.getElementById("clear-logs-btn"),
        
        // Detector Elements
        detectorModelSelect: document.getElementById("detector-model-select"),
        detectorDynamicCheckbox: document.getElementById("detector-dynamic-checkbox"),
        detectorConfRange: document.getElementById("detector-conf-range"),
        confVal: document.getElementById("conf-val"),
        detectorFilterCheckbox: document.getElementById("detector-filter-checkbox"),
        detectorAutoWeatherCheckbox: document.getElementById("detector-auto-weather-checkbox"),
        detectedRain: document.getElementById("detected-rain"),
        detectedFog: document.getElementById("detected-fog"),
        detectedGlare: document.getElementById("detected-glare"),
        detectedHour: document.getElementById("detected-hour"),
        detectedLumMean: document.getElementById("detected-lum-mean"),
        detectedLumStd: document.getElementById("detected-lum-std"),
        activeModelBadge: document.getElementById("active-model-badge"),
        imageDropzone: document.getElementById("image-dropzone"),
        imageFileInput: document.getElementById("image-file-input"),
        runDetectionBtn: document.getElementById("run-detection-btn"),
        visualizerPlaceholder: document.getElementById("visualizer-placeholder"),
        visualizerImg: document.getElementById("visualizer-img"),
        detectionsTable: document.getElementById("detections-table"),
        
        // Trainer Elements
        trainBackboneSelect: document.getElementById("train-backbone-select"),
        trainRoundsInput: document.getElementById("train-rounds-input"),
        trainClientsInput: document.getElementById("train-clients-input"),
        trainPortInput: document.getElementById("train-port-input"),
        trainDummyCheckbox: document.getElementById("train-dummy-checkbox"),
        datasetDropzone: document.getElementById("dataset-dropzone"),
        datasetFileInput: document.getElementById("dataset-file-input"),
        datasetUploadStatus: document.getElementById("dataset-upload-status"),
        startTrainingBtn: document.getElementById("start-training-btn"),
        stopTrainingBtn: document.getElementById("stop-training-btn"),
        roundsProgressPct: document.getElementById("rounds-progress-pct"),
        roundsProgressBar: document.getElementById("rounds-progress-bar"),
        
        // Nodes Status
        nodeServerBadge: document.getElementById("node-server-badge"),
        nodeServerPid: document.getElementById("node-server-pid"),
        nodeClient0Badge: document.getElementById("node-client0-badge"),
        nodeClient0Pid: document.getElementById("node-client0-pid"),
        nodeClient1Badge: document.getElementById("node-client1-badge"),
        nodeClient1Pid: document.getElementById("node-client1-pid"),
        
        // Config Elements
        configButtons: document.querySelectorAll(".config-btn"),
        editorTitle: document.getElementById("editor-title"),
        configTextarea: document.getElementById("config-textarea"),
        saveConfigBtn: document.getElementById("save-config-btn"),
        configSaveStatus: document.getElementById("config-save-status"),
    };

    // ======================================================================
    // Page Tab Navigation
    // ======================================================================
    elements.navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabName = item.getAttribute("data-tab");
            switchTab(tabName);
        });
    });

    function switchTab(tabName) {
        elements.navItems.forEach(nav => nav.classList.remove("active"));
        elements.tabPanels.forEach(panel => panel.classList.remove("active"));
        
        const activeNav = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
        const activePanel = document.getElementById(`tab-${tabName}`);
        
        if (activeNav && activePanel) {
            activeNav.classList.add("active");
            activePanel.classList.add("active");
            appState.activeTab = tabName;
            
            // Update title text dynamically
            if (tabName === "dashboard") {
                elements.pageTitle.textContent = "Federated Learning Control Center";
                elements.pageSubtitle.textContent = "Real-time telemetry and open-world runway debris classification";
            } else if (tabName === "detector") {
                elements.pageTitle.textContent = "Airport Runway FOD Object Visualizer";
                elements.pageSubtitle.textContent = "Run real-time deep learning predictions on test tarmac photos";
            } else if (tabName === "trainer") {
                elements.pageTitle.textContent = "Federated Learning Model Trainer";
                elements.pageSubtitle.textContent = "Configure parameters, upload new dataset files, and trigger client runs";
            } else if (tabName === "configs") {
                elements.pageTitle.textContent = "System Configurations File Manager";
                elements.pageSubtitle.textContent = "Edit model variables and airport deployment settings directly in browser";
                loadConfigFile(appState.currentConfigType);
            }
        }
    }

    // ======================================================================
    // WebSocket System (Real-time Telemetry)
    // ======================================================================
    function setupWebSockets() {
        // --- Stream 1: Training Logs ---
        appState.wsLogs = new WebSocket(`${appState.wsBase}/ws/logs`);
        appState.wsLogs.onmessage = (event) => {
            appendLogLine(event.data);
        };
        appState.wsLogs.onerror = (err) => console.warn("Logs WebSocket error:", err);
        appState.wsLogs.onclose = () => {
            setTimeout(setupWebSockets, 5000); // Reconnect
        };

        // --- Stream 2: Round Metrics ---
        appState.wsMetrics = new WebSocket(`${appState.wsBase}/ws/metrics`);
        appState.wsMetrics.onmessage = (event) => {
            try {
                const metrics = JSON.parse(event.data);
                updateDashboardWithRoundMetrics(metrics);
            } catch (e) {
                console.warn("Metrics parse error:", e);
            }
        };
        appState.wsMetrics.onerror = (err) => console.warn("Metrics WebSocket error:", err);
    }

    function appendLogLine(line) {
        const div = document.createElement("div");
        div.className = "log-line";
        
        if (line.includes("[SYSTEM]") || line.includes("Starting Flower")) {
            div.classList.add("system-msg");
        } else if (line.includes("Error") || line.includes("Failed") || line.includes("Traceback")) {
            div.classList.add("error-msg");
        } else if (line.includes("Round ") && line.includes("mAP@50")) {
            div.classList.add("round-msg");
        } else {
            div.classList.add("info-msg");
        }
        
        div.textContent = line;
        elements.logConsole.appendChild(div);
        elements.logConsole.scrollTop = elements.logConsole.scrollHeight;
    }

    elements.clearLogsBtn.addEventListener("click", () => {
        elements.logConsole.innerHTML = "";
    });

    // ======================================================================
    // Chart.js Configuration
    // ======================================================================
    function initChart() {
        const ctx = document.getElementById('metricsChart').getContext('2d');
        appState.roundsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'mAP@50 (Precision)',
                        data: [],
                        borderColor: '#45f3ff',
                        backgroundColor: 'rgba(69, 243, 255, 0.05)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#9b51e0',
                        backgroundColor: 'rgba(155, 81, 224, 0.05)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8a90a6', font: { family: 'Outfit' } }
                    },
                    y: {
                        position: 'left',
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8a90a6', font: { family: 'Outfit' } },
                        min: 0,
                        max: 1
                    },
                    y1: {
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#8a90a6', font: { family: 'Outfit' } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f5f6f9', font: { family: 'Outfit', size: 12 } }
                    }
                }
            }
        });
    }

    function updateDashboardWithRoundMetrics(metrics) {
        const roundNum = metrics.round;
        const loss = metrics.train_loss || metrics.eval_loss || 0.0;
        const mAP = metrics.mAP50 || metrics.mAP || 0.0;
        
        // Update stats widgets
        elements.currentRoundVal.textContent = roundNum;
        elements.giniVal.textContent = (metrics.gini || 0.0).toFixed(3);
        
        // Update charts datasets
        if (appState.roundsChart) {
            const chart = appState.roundsChart;
            if (!chart.data.labels.includes(`Round ${roundNum}`)) {
                chart.data.labels.push(`Round ${roundNum}`);
                chart.data.datasets[0].data.push(mAP);
                chart.data.datasets[1].data.push(loss);
                chart.update();
            }
        }
    }

    // ======================================================================
    // FOD Detector Module
    // ======================================================================
    // Sliders
    elements.detectorConfRange.addEventListener("input", (e) => {
        elements.confVal.textContent = e.target.value;
    });

    // Toggle weather sliders visibility
    elements.detectorFilterCheckbox.addEventListener("change", (e) => {
        const group = document.getElementById("weather-parameters-group");
        if (e.target.checked) {
            group.classList.add("hidden");
        } else {
            group.classList.remove("hidden");
        }
    });

    // Toggle dynamic mode checkbox
    elements.detectorDynamicCheckbox.addEventListener("change", (e) => {
        elements.detectorModelSelect.disabled = e.target.checked;
    });

    // Toggle auto weather checkbox
    elements.detectorAutoWeatherCheckbox.addEventListener("change", (e) => {
        const manualSliders = document.getElementById("manual-weather-sliders");
        const detectedInfo = document.getElementById("detected-weather-info");
        if (e.target.checked) {
            manualSliders.classList.add("hidden");
            detectedInfo.classList.remove("hidden");
        } else {
            manualSliders.classList.remove("hidden");
            detectedInfo.classList.add("hidden");
        }
    });

    // Weather Sliders values bindings
    document.getElementById("detector-rain-range").addEventListener("input", (e) => {
        document.getElementById("rain-val").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("detector-fog-range").addEventListener("input", (e) => {
        document.getElementById("fog-val").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("detector-glare-range").addEventListener("input", (e) => {
        document.getElementById("glare-val").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("detector-hour-range").addEventListener("input", (e) => {
        document.getElementById("hour-val").textContent = `${e.target.value}:00`;
    });

    // Dropzone logic
    setupDragDrop(elements.imageDropzone, elements.imageFileInput, (file) => {
        appState.selectedImageFile = file;
        elements.runDetectionBtn.disabled = false;
        
        // Preview selected file in dropzone
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.visualizerPlaceholder.style.display = "none";
            elements.visualizerImg.classList.remove("hidden");
            elements.visualizerImg.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });

    // Click trigger on Dropzone
    elements.imageDropzone.addEventListener("click", () => {
        elements.imageFileInput.click();
    });

    elements.runDetectionBtn.addEventListener("click", async () => {
        if (!appState.selectedImageFile) return;

        elements.runDetectionBtn.disabled = true;
        elements.runDetectionBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing prediction...`;

        const formData = new FormData();
        formData.append("file", appState.selectedImageFile);
        formData.append("model_name", elements.detectorModelSelect.value);
        formData.append("conf", elements.detectorConfRange.value);
        formData.append("no_filter", elements.detectorFilterCheckbox.checked ? "true" : "false");
        formData.append("rain_prob", document.getElementById("detector-rain-range").value);
        formData.append("fog_prob", document.getElementById("detector-fog-range").value);
        formData.append("glare_prob", document.getElementById("detector-glare-range").value);
        formData.append("hour", document.getElementById("detector-hour-range").value);
        formData.append("dynamic_mode", elements.detectorDynamicCheckbox.checked ? "true" : "false");
        formData.append("auto_weather", elements.detectorAutoWeatherCheckbox.checked ? "true" : "false");

        try {
            const res = await fetch(`${appState.apiBase}/api/predict`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            
            // Show annotated image
            elements.visualizerImg.src = data.image_base64;
            
            // Render table results
            renderDetectionsTable(data.detections);

            // Update active model badge
            if (data.active_model) {
                const modelName = data.active_model.replace("checkpoints/", "").replace(".pt", "").toUpperCase();
                elements.activeModelBadge.textContent = `Model: ${modelName}`;
                elements.activeModelBadge.style.display = "flex";
            } else {
                elements.activeModelBadge.style.display = "none";
            }

            // Update auto-detected weather indicators
            if (data.auto_weather_detected && elements.detectorAutoWeatherCheckbox.checked) {
                const w = data.auto_weather_detected;
                elements.detectedRain.textContent = `${(w.rain_prob * 100).toFixed(0)}%`;
                elements.detectedFog.textContent = `${(w.fog_prob * 100).toFixed(0)}%`;
                elements.detectedGlare.textContent = `${(w.glare_prob * 100).toFixed(0)}%`;
                elements.detectedHour.textContent = `${Math.floor(w.hour)}:00`;
                elements.detectedLumMean.textContent = w.luminance_mean.toFixed(2);
                elements.detectedLumStd.textContent = w.luminance_std.toFixed(2);
            } else {
                elements.detectedRain.textContent = "N/A";
                elements.detectedFog.textContent = "N/A";
                elements.detectedGlare.textContent = "N/A";
                elements.detectedHour.textContent = "N/A";
                elements.detectedLumMean.textContent = "N/A";
                elements.detectedLumStd.textContent = "N/A";
            }
        } catch (e) {
            alert(`Prediction Failed: ${e.message}`);
        } finally {
            elements.runDetectionBtn.disabled = false;
            elements.runDetectionBtn.innerHTML = `<i class="fa-solid fa-circle-play"></i> Run FOD Analysis`;
        }
    });

    function renderDetectionsTable(detections) {
        const tbody = elements.detectionsTable.querySelector("tbody");
        tbody.innerHTML = "";
        
        if (!detections || detections.length === 0) {
            tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No FOD objects detected. Tarmac is clean!</td></tr>`;
            return;
        }

        detections.forEach(det => {
            const tr = document.createElement("tr");
            const boxStr = `[${det.bbox.join(", ")}]`;
            tr.innerHTML = `
                <td>${det.id}</td>
                <td><strong>${det.class_name}</strong></td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td><code>${boxStr}</code></td>
                <td><span class="badge port-badge" style="border-radius:4px; font-size:11px; padding:3px 6px;">${det.clip_label}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // ======================================================================
    // Model Trainer Module
    // ======================================================================
    setupDragDrop(elements.datasetDropzone, elements.datasetFileInput, async (file) => {
        elements.datasetUploadStatus.className = "upload-status-msg system-msg";
        elements.datasetUploadStatus.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Uploading dataset zip...`;
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${appState.apiBase}/api/dataset/upload`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            
            elements.datasetUploadStatus.className = "upload-status-msg success-msg";
            elements.datasetUploadStatus.style.color = "var(--accent-green)";
            elements.datasetUploadStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.message}`;
        } catch (e) {
            elements.datasetUploadStatus.className = "upload-status-msg error-msg";
            elements.datasetUploadStatus.style.color = "var(--accent-red)";
            elements.datasetUploadStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Fail: ${e.message}`;
        }
    });

    elements.datasetDropzone.addEventListener("click", () => {
        elements.datasetFileInput.click();
    });

    elements.startTrainingBtn.addEventListener("click", async () => {
        const body = {
            rounds: parseInt(elements.trainRoundsInput.value),
            min_clients: parseInt(elements.trainClientsInput.value),
            port: parseInt(elements.trainPortInput.value),
            dummy_model: elements.trainDummyCheckbox.checked,
            backbone: elements.trainBackboneSelect.value,
            clients: [] // Empty defaults to automatic setup
        };

        elements.startTrainingBtn.disabled = true;
        
        try {
            const res = await fetch(`${appState.apiBase}/api/training/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });

            if (!res.ok) throw new Error(await res.text());
            
            appendLogLine("[SYSTEM] Federated training initiated successfully.");
            pollTrainingState();
        } catch (e) {
            alert(`Start training failed: ${e.message}`);
            elements.startTrainingBtn.disabled = false;
        }
    });

    elements.stopTrainingBtn.addEventListener("click", async () => {
        elements.stopTrainingBtn.disabled = true;
        try {
            const res = await fetch(`${appState.apiBase}/api/training/stop`, {
                method: "POST"
            });
            if (!res.ok) throw new Error(await res.text());
            appendLogLine("[SYSTEM] Stop command sent to FL nodes.");
        } catch (e) {
            alert(`Stop training failed: ${e.message}`);
        }
    });

    // ======================================================================
    // Configurations Tab Module
    // ======================================================================
    elements.configButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            elements.configButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const fileType = btn.getAttribute("data-file");
            appState.currentConfigType = fileType;
            loadConfigFile(fileType);
        });
    });

    async function loadConfigFile(fileType) {
        let endpoint = `${appState.apiBase}/api/config/global`;
        if (fileType !== "global") {
            endpoint = `${appState.apiBase}/api/config/airports/${fileType}`;
        }
        
        elements.editorTitle.textContent = `Editing ${fileType === 'global' ? 'global_config.yaml' : fileType + '.yaml'}`;
        elements.configTextarea.value = "Loading config data...";

        try {
            const res = await fetch(endpoint);
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            
            // Format yaml cleanly in text area
            if (yamlStringify) {
                elements.configTextarea.value = yamlStringify(data);
            } else {
                elements.configTextarea.value = JSON.stringify(data, null, 2);
            }
        } catch (e) {
            elements.configTextarea.value = `Failed to load config: ${e.message}`;
        }
    }

    elements.saveConfigBtn.addEventListener("click", async () => {
        const yamlText = elements.configTextarea.value;
        let parsedData = null;
        
        // Parse textarea yaml text back to JSON object
        try {
            parsedData = yamlParse(yamlText);
        } catch (e) {
            elements.configSaveStatus.style.color = "var(--accent-red)";
            elements.configSaveStatus.textContent = `YAML Parse Error: ${e.message}`;
            return;
        }

        elements.saveConfigBtn.disabled = true;
        elements.configSaveStatus.style.color = "var(--accent-cyan)";
        elements.configSaveStatus.textContent = "Saving config changes...";

        let endpoint = `${appState.apiBase}/api/config/global`;
        if (appState.currentConfigType !== "global") {
            endpoint = `${appState.apiBase}/api/config/airports/${appState.currentConfigType}`;
        }

        try {
            const res = await fetch(endpoint, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(parsedData)
            });

            if (!res.ok) throw new Error(await res.text());
            
            elements.configSaveStatus.style.color = "var(--accent-green)";
            elements.configSaveStatus.textContent = "Config changes saved and active!";
            setTimeout(() => { elements.configSaveStatus.textContent = ""; }, 4000);
        } catch (e) {
            elements.configSaveStatus.style.color = "var(--accent-red)";
            elements.configSaveStatus.textContent = `Save Failed: ${e.message}`;
        } finally {
            elements.saveConfigBtn.disabled = false;
        }
    });

    // ======================================================================
    // System Helpers & Drag Drop Setup
    // ======================================================================
    function setupDragDrop(zone, input, onFileSelected) {
        input.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                onFileSelected(e.target.files[0]);
            }
        });

        zone.addEventListener("dragover", (e) => {
            e.preventDefault();
            zone.classList.add("drag-active");
        });

        zone.addEventListener("dragleave", () => {
            zone.classList.remove("drag-active");
        });

        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            zone.classList.remove("drag-active");
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                onFileSelected(e.dataTransfer.files[0]);
            }
        });
    }

    // Polling Training State
    async function pollTrainingState() {
        try {
            const res = await fetch(`${appState.apiBase}/api/training/state`);
            if (!res.ok) return;
            const state = await res.json();
            
            updateTrainerTelemetryUI(state);
            
            if (state.phase === "running" || state.phase === "starting" || state.phase === "stopping") {
                appState.trainingActive = true;
                setTimeout(pollTrainingState, 1500); // Poll every 1.5s
            } else {
                appState.trainingActive = false;
                elements.startTrainingBtn.disabled = false;
                elements.stopTrainingBtn.disabled = true;
            }
        } catch (e) {
            console.warn("Telemetry poll error:", e);
            setTimeout(pollTrainingState, 3000);
        }
    }

    function updateTrainerTelemetryUI(state) {
        // Status indicator
        elements.trainingStatusBadge.className = `badge training-badge ${state.phase}`;
        
        let statusText = "Training Idle";
        if (state.phase === "starting") statusText = "FL Starting...";
        if (state.phase === "running") statusText = `FL Round ${state.current_round}/${state.total_rounds}`;
        if (state.phase === "stopping") statusText = "FL Stopping...";
        if (state.phase === "completed") statusText = "FL Completed";
        if (state.phase === "failed") statusText = "FL Failed";
        
        elements.trainingStatusBadge.querySelector("span").textContent = statusText;
        
        // Progress bar
        elements.totalRoundsVal.textContent = state.total_rounds;
        if (state.total_rounds > 0) {
            const pct = Math.round((state.current_round / state.total_rounds) * 100);
            elements.roundsProgressPct.textContent = `${pct}%`;
            elements.roundsProgressBar.style.width = `${pct}%`;
        } else {
            elements.roundsProgressPct.textContent = "0%";
            elements.roundsProgressBar.style.width = "0%";
        }

        // Action Buttons
        if (state.phase === "running" || state.phase === "starting") {
            elements.startTrainingBtn.disabled = true;
            elements.stopTrainingBtn.disabled = false;
        } else {
            elements.startTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = true;
        }

        // Active node cards
        updateNodeCard(elements.nodeServerBadge, elements.nodeServerPid, state.server_pid);
        updateNodeCard(elements.nodeClient0Badge, elements.nodeClient0Pid, state.client_pids[0]);
        updateNodeCard(elements.nodeClient1Badge, elements.nodeClient1Pid, state.client_pids[1]);
        
        // Update general dashboard items
        elements.activeClientsVal.textContent = state.connected_clients;
        elements.uptimeVal.textContent = `${Math.round(state.elapsed_seconds)}s`;
    }

    function updateNodeCard(badge, pidText, pid) {
        if (pid) {
            badge.className = "node-badge running";
            badge.textContent = "Active";
            pidText.textContent = pid;
        } else {
            badge.className = "node-badge";
            badge.textContent = "Offline";
            pidText.textContent = "N/A";
        }
    }

    // Helper: Simple client-side YAML parsing & stringifying using js-yaml
    function yamlParse(text) {
        if (typeof jsyaml !== 'undefined') {
            return jsyaml.load(text);
        }
        return JSON.parse(text);
    }
    
    function yamlStringify(obj) {
        if (typeof jsyaml !== 'undefined') {
            return jsyaml.dump(obj);
        }
        return JSON.stringify(obj, null, 2);
    }

    async function loadCheckpoints() {
        try {
            const res = await fetch(`${appState.apiBase}/api/checkpoints`);
            if (res.ok) {
                const checkpoints = await res.json();
                checkpoints.forEach(cp => {
                    const option = document.createElement("option");
                    option.value = cp;
                    const name = cp.replace("checkpoints/", "").replace(".pt", "");
                    option.textContent = `FedFOD Checkpoint ${name.replace("_", " ").toUpperCase()}`;
                    elements.detectorModelSelect.appendChild(option);
                });
                if (checkpoints.length > 0) {
                    elements.detectorModelSelect.value = checkpoints[0];
                }
            }
        } catch (e) {
            console.warn("Failed to load checkpoints:", e);
        }
    }

    // Startup
    initChart();
    setupWebSockets();
    pollTrainingState();
    loadCheckpoints();
});
