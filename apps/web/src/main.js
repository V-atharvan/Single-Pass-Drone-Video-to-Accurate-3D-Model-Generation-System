/**
 * AEROMAP 3D — Core Application Controller & View Router
 */
import { Drone3DViewer } from './components/viewer3d.js';

class AppController {
  constructor() {
    this.currentView = 'dashboard';
    this.viewerInstance = null;

    // Sample State
    this.projects = [
      {
        id: 'proj-101',
        name: 'Coastal Infrastructure & Port Survey v4.2',
        location: 'San Diego Harbor, CA',
        crs: 'EPSG:32611 (UTM 11N)',
        flightCount: 4,
        modelCount: 2,
        qualityScore: 94,
        date: 'Sept 07, 2026',
        pointCount: '18.4M',
        thumbnail: '/assets/hero_3d_model.jpg',
      },
      {
        id: 'proj-102',
        name: 'Downtown Metro Corridor Digital Twin',
        location: 'Austin, TX',
        crs: 'EPSG:32614 (UTM 14N)',
        flightCount: 2,
        modelCount: 1,
        qualityScore: 91,
        date: 'Sept 05, 2026',
        pointCount: '12.1M',
        thumbnail: '/assets/hero_3d_model.jpg',
      },
      {
        id: 'proj-103',
        name: 'High-Altitude Alpine Ridge Slope Survey',
        location: 'Vail Pass, CO',
        crs: 'EPSG:32613 (UTM 13N)',
        flightCount: 1,
        modelCount: 1,
        qualityScore: 89,
        date: 'Aug 29, 2026',
        pointCount: '9.8M',
        thumbnail: '/assets/hero_3d_model.jpg',
      },
    ];

    this.activeJob = {
      id: 'job-9842',
      projectName: 'Coastal Infrastructure & Port Survey v4.2',
      stage: 'ESTIMATING_DEPTH',
      stageDisplay: 'Monocular AI Depth Estimation (Depth Anything V2)',
      progress: 78,
      framesProcessed: 8421,
      totalFrames: 11672,
      fps: 32.4,
      gpuVram: '5.8 / 24 GB',
      eta: '3m 48s',
      positioningMode: 'RTK_PPK',
    };

    this.initNavigation();
    this.renderView(this.currentView);
  }

  initNavigation() {
    // Desktop Sidebar Click Handlers
    document.querySelectorAll('.app-sidebar .nav-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        if (view) this.switchView(view);
      });
    });

    // Mobile Bottom Nav Click Handlers
    document.querySelectorAll('.mobile-bottom-nav .mobile-nav-item').forEach((btn) => {
      if (btn.id === 'btn-mobile-more') return;
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        if (view) this.switchView(view);
      });
    });

    // Mobile "More" Drawer Toggle
    const btnMore = document.getElementById('btn-mobile-more');
    const drawerOverlay = document.getElementById('mobile-drawer-overlay');
    const btnDrawerClose = document.getElementById('btn-drawer-close');

    if (btnMore && drawerOverlay) {
      btnMore.addEventListener('click', () => {
        drawerOverlay.classList.add('open');
      });
      if (btnDrawerClose) {
        btnDrawerClose.addEventListener('click', () => {
          drawerOverlay.classList.remove('open');
        });
      }
      drawerOverlay.addEventListener('click', (e) => {
        if (e.target === drawerOverlay) drawerOverlay.classList.remove('open');
      });

      // Drawer Destination Buttons
      drawerOverlay.querySelectorAll('.drawer-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          const view = btn.dataset.view;
          if (view) {
            drawerOverlay.classList.remove('open');
            this.switchView(view);
          }
        });
      });
    }

    // Header Quick Upload button
    const btnQuickUpload = document.getElementById('btn-quick-upload');
    if (btnQuickUpload) {
      btnQuickUpload.addEventListener('click', () => this.switchView('upload'));
    }

    // Header Pipeline Pill click
    const headerPill = document.getElementById('header-pipeline-pill');
    if (headerPill) {
      headerPill.addEventListener('click', () => this.switchView('processing'));
    }
  }

  switchView(viewName) {
    this.currentView = viewName;

    // Update active classes on desktop sidebar
    document.querySelectorAll('.app-sidebar .nav-item').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    // Update active classes on mobile bottom nav
    document.querySelectorAll('.mobile-bottom-nav .mobile-nav-item').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    this.renderView(viewName);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  renderView(viewName) {
    const main = document.getElementById('main-content');
    if (!main) return;

    if (viewName === 'dashboard') {
      main.innerHTML = this.renderDashboard();
      this.attachDashboardEvents();
    } else if (viewName === 'projects') {
      main.innerHTML = this.renderProjects();
      this.attachProjectsEvents();
    } else if (viewName === 'upload') {
      main.innerHTML = this.renderUpload();
      this.attachUploadEvents();
    } else if (viewName === 'processing') {
      main.innerHTML = this.renderProcessing();
      this.attachProcessingEvents();
    } else if (viewName === 'viewer') {
      main.innerHTML = this.renderViewer();
      this.attachViewerEvents();
    } else if (viewName === 'exports') {
      main.innerHTML = this.renderExports();
      this.attachExportsEvents();
    } else if (viewName === 'settings') {
      main.innerHTML = this.renderSettings();
    }
  }

  /* ========================================================================
     1. Dashboard View
     ======================================================================== */
  renderDashboard() {
    return `
      <div class="view-header">
        <div>
          <h1 class="view-headline">Operations Dashboard</h1>
          <p class="view-subtitle">Real-time status of single-pass drone reconstructions and 3D digital twins.</p>
        </div>
        <button class="btn-primary-action" id="btn-dash-new-survey">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span>Start Reconstruction</span>
        </button>
      </div>

      <!-- Key Performance Metrics -->
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-title">Surveyed Area</div>
          <div class="metric-value-row">
            <span class="metric-number">4.2</span>
            <span class="metric-unit">km²</span>
          </div>
          <div class="metric-sub">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="18 15 12 9 6 15"></polyline></svg>
            <span>+1.1 km² this week</span>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Reconstructed Points</div>
          <div class="metric-value-row">
            <span class="metric-number">40.3</span>
            <span class="metric-unit">Million</span>
          </div>
          <div class="metric-sub">
            <span>Avg 820 pts/m² density</span>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Horizontal RMSE</div>
          <div class="metric-value-row">
            <span class="metric-number">2.4</span>
            <span class="metric-unit">cm</span>
          </div>
          <div class="metric-sub">
            <span>RTK/PPK tight fusion</span>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Active Workers</div>
          <div class="metric-value-row">
            <span class="metric-number">4</span>
            <span class="metric-unit">NVIDIA GPUs</span>
          </div>
          <div class="metric-sub">
            <span>CUDA 12.2 • FP16 Autocast</span>
          </div>
        </div>
      </div>

      <!-- Live Pipeline Processing Card -->
      <div class="active-job-banner">
        <div class="job-banner-header">
          <div class="job-title-group">
            <span class="job-status-badge">PROCESSING</span>
            <h2 class="job-title">${this.activeJob.projectName}</h2>
          </div>
          <button class="btn-open-3d" id="btn-dash-view-live" style="max-width: 180px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span>Live Monitor</span>
          </button>
        </div>

        <div class="job-progress-row">
          <div class="job-progress-bar-large">
            <div class="job-progress-bar-fill" style="width: ${this.activeJob.progress}%;"></div>
          </div>
          <span class="job-pct-large">${this.activeJob.progress}%</span>
        </div>

        <div class="job-meta-grid">
          <div class="job-meta-item">
            <span class="meta-k">CURRENT STAGE</span>
            <span class="meta-v" style="color: var(--accent-cyan);">${this.activeJob.stageDisplay}</span>
          </div>
          <div class="job-meta-item">
            <span class="meta-k">FRAMES PROCESSED</span>
            <span class="meta-v">${this.activeJob.framesProcessed.toLocaleString()} / ${this.activeJob.totalFrames.toLocaleString()}</span>
          </div>
          <div class="job-meta-item">
            <span class="meta-k">PROCESSING SPEED</span>
            <span class="meta-v">${this.activeJob.fps} FPS</span>
          </div>
          <div class="job-meta-item">
            <span class="meta-k">POSITIONING MODE</span>
            <span class="meta-v" style="color: var(--accent-green);">${this.activeJob.positioningMode}</span>
          </div>
          <div class="job-meta-item">
            <span class="meta-k">ESTIMATED TIME REMAINING</span>
            <span class="meta-v">${this.activeJob.eta}</span>
          </div>
        </div>
      </div>

      <!-- Recent Projects Grid -->
      <div class="section-title-row">
        <h2 class="section-title">Recent 3D Survey Projects</h2>
        <span style="color: var(--text-muted); font-size: 0.85rem;">Showing ${this.projects.length} completed models</span>
      </div>

      <div class="projects-grid">
        ${this.projects.map((p) => `
          <div class="project-card" data-proj-id="${p.id}">
            <div class="project-thumbnail" style="background-image: url('${p.thumbnail}')">
              <span class="card-badge">${p.crs}</span>
              <span class="card-badge" style="color: var(--accent-green); border-color: rgba(0,230,118,0.3)">READY</span>
            </div>
            <div class="project-content">
              <h3 class="project-card-title">${p.name}</h3>
              <div class="project-card-loc">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
                <span>${p.location}</span>
              </div>
              <div class="project-metrics-row">
                <div class="p-stat">
                  <span class="p-stat-label">DENSITY</span>
                  <span class="p-stat-val">${p.pointCount}</span>
                </div>
                <div class="p-stat">
                  <span class="p-stat-label">QUALITY</span>
                  <span class="p-stat-val" style="color: var(--accent-green)">${p.qualityScore}%</span>
                </div>
                <div class="p-stat">
                  <span class="p-stat-label">FLIGHTS</span>
                  <span class="p-stat-val">${p.flightCount} passes</span>
                </div>
              </div>
              <div class="project-card-actions">
                <button class="btn-open-3d btn-card-open-3d" data-id="${p.id}">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                  <span>Open in 3D Viewer</span>
                </button>
              </div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  attachDashboardEvents() {
    const btnDashLive = document.getElementById('btn-dash-view-live');
    if (btnDashLive) {
      btnDashLive.addEventListener('click', () => this.switchView('processing'));
    }

    const btnNewSurvey = document.getElementById('btn-dash-new-survey');
    if (btnNewSurvey) {
      btnNewSurvey.addEventListener('click', () => this.switchView('upload'));
    }

    document.querySelectorAll('.btn-card-open-3d').forEach((btn) => {
      btn.addEventListener('click', () => this.switchView('viewer'));
    });
  }

  /* ========================================================================
     2. Projects View
     ======================================================================== */
  renderProjects() {
    return `
      <div class="view-header">
        <div>
          <h1 class="view-headline">Project Digital Twins</h1>
          <p class="view-subtitle">Organize flights, inspection models, and georeferenced survey datasets.</p>
        </div>
        <button class="btn-primary-action" id="btn-modal-create-proj">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span>New Project</span>
        </button>
      </div>

      <div class="projects-grid">
        ${this.projects.map((p) => `
          <div class="project-card">
            <div class="project-thumbnail" style="background-image: url('${p.thumbnail}')">
              <span class="card-badge">${p.crs}</span>
              <span class="card-badge" style="color: var(--accent-cyan); border-color: rgba(0,240,255,0.3)">${p.date}</span>
            </div>
            <div class="project-content">
              <h3 class="project-card-title">${p.name}</h3>
              <div class="project-card-loc">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
                <span>${p.location}</span>
              </div>
              <div class="project-metrics-row">
                <div class="p-stat">
                  <span class="p-stat-label">POINTS</span>
                  <span class="p-stat-val">${p.pointCount}</span>
                </div>
                <div class="p-stat">
                  <span class="p-stat-label">QUALITY SCORE</span>
                  <span class="p-stat-val" style="color: var(--accent-green)">${p.qualityScore}%</span>
                </div>
                <div class="p-stat">
                  <span class="p-stat-label">FLIGHT PASSES</span>
                  <span class="p-stat-val">${p.flightCount}</span>
                </div>
              </div>
              <div class="project-card-actions">
                <button class="btn-open-3d btn-card-open-3d">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                  <span>Inspect 3D</span>
                </button>
              </div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  attachProjectsEvents() {
    const btnCreate = document.getElementById('btn-modal-create-proj');
    if (btnCreate) {
      btnCreate.addEventListener('click', () => {
        const name = prompt('Enter New Project Name:', 'Solar Farm Photogrammetry Block C');
        if (name) {
          this.projects.unshift({
            id: 'proj-' + Date.now(),
            name,
            location: 'Nevada Desert, NV',
            crs: 'EPSG:32611 (UTM 11N)',
            flightCount: 1,
            modelCount: 0,
            qualityScore: 92,
            date: 'Today',
            pointCount: 'Pending',
            thumbnail: '/assets/hero_3d_model.jpg',
          });
          this.renderView('projects');
        }
      });
    }

    document.querySelectorAll('.btn-card-open-3d').forEach((btn) => {
      btn.addEventListener('click', () => this.switchView('viewer'));
    });
  }

  /* ========================================================================
     3. Upload View & Pre-Flight Assessment
     ======================================================================== */
  renderUpload() {
    return `
      <div class="upload-container">
        <div class="view-header">
          <div>
            <h1 class="view-headline">Upload Drone Video & Telemetry</h1>
            <p class="view-subtitle">Direct-to-S3 multi-part streaming for 4K video and GPS/IMU companion logs.</p>
          </div>
        </div>

        <!-- Dropzone Card -->
        <div class="dropzone-card" id="dropzone">
          <input type="file" id="file-video" accept="video/mp4,video/quicktime,video/x-matroska" style="display: none;">
          <div class="dropzone-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          </div>
          <h3 class="dropzone-title">Drag & Drop Drone Video Here</h3>
          <p class="dropzone-sub">Supports 4K / 1080p MP4, MOV, or MKV up to 50 GB. Companion SRT, CSV, or JSON telemetry files are automatically matched.</p>
          <button class="btn-browse" id="btn-browse-files">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
            <span>Select Video File</span>
          </button>
        </div>

        <!-- Pre-Flight Quality Report (Instant Feedback HUD) -->
        <div class="quality-report-card">
          <div class="quality-score-header">
            <div>
              <h3 style="font-family: var(--font-display); font-size: 1.15rem; color: #FFFFFF;">Pre-Flight Quality Score</h3>
              <p style="color: var(--text-secondary); font-size: 0.8rem;">Automated validation from TASK-016 & TASK-022 before GPU launch.</p>
            </div>
            <div class="quality-score-pill">
              <span style="font-size: 1.5rem; line-height: 1;">94</span>
              <span style="font-size: 0.85rem; font-family: var(--font-mono);">/ 100 • HIGH QUALITY</span>
            </div>
          </div>

          <div class="quality-subscores-grid">
            <div class="subscore-card">
              <div class="subscore-name">VISUAL CLARITY</div>
              <div class="subscore-val" style="color: var(--accent-green)">95%</div>
              <span style="font-size: 0.72rem; color: var(--text-muted);">Sharp building facades</span>
            </div>

            <div class="subscore-card">
              <div class="subscore-name">GPS DILUTION (HDOP)</div>
              <div class="subscore-val" style="color: var(--accent-cyan)">0.78</div>
              <span style="font-size: 0.72rem; color: var(--text-muted);">RTK precision fix</span>
            </div>

            <div class="subscore-card">
              <div class="subscore-name">MOTION BLUR</div>
              <div class="subscore-val" style="color: var(--accent-green)">Low</div>
              <span style="font-size: 0.72rem; color: var(--text-muted);">&lt; 3% degraded frames</span>
            </div>

            <div class="subscore-card">
              <div class="subscore-name">ESTIMATED ACCURACY</div>
              <div class="subscore-val" style="color: var(--accent-cyan)">&plusmn; 2.5 cm</div>
              <span style="font-size: 0.72rem; color: var(--text-muted);">Surpasses 5cm target</span>
            </div>
          </div>

          <!-- Configuration options -->
          <div style="margin-bottom: 24px;">
            <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; color: var(--text-muted); margin-bottom: 10px;">RECONSTRUCTION QUALITY PRESET</div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
              <button class="viewer-btn-mode active" style="padding: 8px 16px; font-size: 0.85rem;">Balanced (Recommended)</button>
              <button class="viewer-btn-mode" style="padding: 8px 16px; font-size: 0.85rem;">Preview (Fast, 5m)</button>
              <button class="viewer-btn-mode" style="padding: 8px 16px; font-size: 0.85rem;">Maximum Detail (Sub-cm)</button>
            </div>
          </div>

          <button class="btn-primary-action" id="btn-start-reconstruct" style="width: 100%; justify-content: center; padding: 14px; font-size: 1rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            <span>Launch 3D Reconstruction Pipeline</span>
          </button>
        </div>
      </div>
    `;
  }

  attachUploadEvents() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-video');
    const btnBrowse = document.getElementById('btn-browse-files');
    const btnStart = document.getElementById('btn-start-reconstruct');

    if (btnBrowse && fileInput) {
      btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-active');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-active');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-active');
        if (e.dataTransfer.files.length) {
          alert(`Selected: ${e.dataTransfer.files[0].name} (Pre-flight validation passed)`);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
          alert(`Selected: ${fileInput.files[0].name} (Pre-flight validation passed)`);
        }
      });
    }

    if (btnStart) {
      btnStart.addEventListener('click', () => {
        this.switchView('processing');
      });
    }
  }

  /* ========================================================================
     4. Processing View (Live Pipeline Monitor)
     ======================================================================== */
  renderProcessing() {
    return `
      <div style="max-width: 860px; margin: 0 auto; width: 100%;">
        <div class="view-header">
          <div>
            <h1 class="view-headline">Live Reconstruction Pipeline</h1>
            <p class="view-subtitle">Processing single-pass flight through asynchronous GPU stages.</p>
          </div>
          <button class="btn-open-3d" id="btn-proc-goto-3d">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            <span>View 3D Live Feed</span>
          </button>
        </div>

        <!-- Pipeline Stepper -->
        <div class="pipeline-stepper">
          <div class="step-card done">
            <div class="step-icon-wrap">✓</div>
            <div class="step-info">
              <div class="step-name">1. Input Validation & Telemetry Alignment</div>
              <div class="step-desc">11,672 frames validated • DJI SRT synchronized with GPS/IMU timecodes.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-green);">100%</span>
          </div>

          <div class="step-card done">
            <div class="step-icon-wrap">✓</div>
            <div class="step-info">
              <div class="step-name">2. Keyframe Selection & Disk Caching (TASK-024)</div>
              <div class="step-desc">Selected 420 optimal non-blurry keyframes meeting 75% visual overlap.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-green);">100%</span>
          </div>

          <div class="step-card done">
            <div class="step-icon-wrap">✓</div>
            <div class="step-info">
              <div class="step-name">3. Visual Feature Tracking & Epipolar Pose (TASK-026)</div>
              <div class="step-desc">DIS optical flow + FAST/ORB tracking across sequence; 5,200 feature tracks.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-green);">100%</span>
          </div>

          <div class="step-card done">
            <div class="step-info">
              <div class="step-name">4. Sensor Fusion EKF & Factor Graph (TASK-027)</div>
              <div class="step-desc">SciPy LM factor graph converged; RTK/PPK mode active (RMSE: 2.4 cm).</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-green);">100%</span>
          </div>

          <div class="step-card active">
            <div class="step-icon-wrap">●</div>
            <div class="step-info">
              <div class="step-name" style="color: var(--accent-cyan);">5. Monocular AI Depth & Metric Scaling (TASK-031 - 034)</div>
              <div class="step-desc">Inference on Depth Anything V2 with FP16 autocast; edge-preserving bilateral upsampling.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan);">78%</span>
          </div>

          <div class="step-card pending">
            <div class="step-icon-wrap">○</div>
            <div class="step-info">
              <div class="step-name">6. Semantic Scene Understanding & Dynamic Removal (TASK-035)</div>
              <div class="step-desc">SegFormer segmentation with ANIMAL & DYNAMIC_OBJECT exclusion masks.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">QUEUED</span>
          </div>

          <div class="step-card pending">
            <div class="step-icon-wrap">○</div>
            <div class="step-info">
              <div class="step-name">7. Dense Neural Scene Fusion & TSDF Integration</div>
              <div class="step-desc">Voxel grid TSDF truncation with confidence weighting.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">QUEUED</span>
          </div>

          <div class="step-card pending">
            <div class="step-icon-wrap">○</div>
            <div class="step-info">
              <div class="step-name">8. Mesh Extraction & OGC 3D Tiles 1.1 Export</div>
              <div class="step-desc">Screened Poisson reconstruction with texture atlas baking.</div>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">QUEUED</span>
          </div>
        </div>
      </div>
    `;
  }

  attachProcessingEvents() {
    const btnGoto3D = document.getElementById('btn-proc-goto-3d');
    if (btnGoto3D) {
      btnGoto3D.addEventListener('click', () => this.switchView('viewer'));
    }
  }

  /* ========================================================================
     5. 3D Workspace View (Interactive WebGL Canvas)
     ======================================================================== */
  renderViewer() {
    return `
      <div class="viewer-container" id="viewer-container">
        <!-- Top HUD -->
        <div class="viewer-top-hud">
          <div class="viewer-project-pill">
            <span class="viewer-project-title">San Diego Harbor Coastal Twin</span>
            <span class="viewer-telemetry-badge">ALT: 154m • RTK FIXED • 18.4M PTS</span>
          </div>

          <div class="viewer-mode-selectors">
            <button class="viewer-btn-mode active" data-mode="mesh">3D Mesh</button>
            <button class="viewer-btn-mode" data-mode="points">Point Cloud</button>
            <button class="viewer-btn-mode" data-mode="heatmap">Elevation</button>
            <button class="viewer-btn-mode" data-mode="confidence">Confidence</button>
            <button class="viewer-btn-mode" data-mode="wireframe">Wireframe</button>
          </div>
        </div>

        <!-- Sidebar Floating Action Buttons -->
        <div class="viewer-sidebar-controls">
          <button class="viewer-control-btn active" id="btn-toggle-traj" title="Toggle Drone Flight Trajectory (TASK-029)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          </button>
          <button class="viewer-control-btn" id="btn-toggle-rotate" title="Toggle Auto-Orbit Rotation">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
          </button>
          <button class="viewer-control-btn" id="btn-reset-cam" title="Reset Camera View">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          </button>
        </div>

        <!-- Bottom Legend & Measurement Readout -->
        <div class="viewer-bottom-hud">
          <div class="hud-legend">
            <span class="legend-title">Observation State (PRD §6)</span>
            <div class="legend-items">
              <div class="legend-item"><span class="legend-dot dot-observed"></span> OBSERVED</div>
              <div class="legend-item"><span class="legend-dot dot-partial"></span> PARTIAL</div>
              <div class="legend-item"><span class="legend-dot dot-inferred"></span> INFERRED</div>
              <div class="legend-item"><span class="legend-dot dot-trajectory"></span> FLIGHT PATH</div>
            </div>
          </div>

          <div class="hud-measure-readout">
            <div class="measure-metric">
              <span class="measure-lbl">HORIZONTAL DISTANCE</span>
              <span class="measure-val" id="measure-dist">124.8 m</span>
            </div>
            <div class="measure-metric">
              <span class="measure-lbl">ELEVATION DELTA (&Delta;Z)</span>
              <span class="measure-val" id="measure-z">18.2 m</span>
            </div>
            <div class="measure-metric">
              <span class="measure-lbl">CONFIDENCE</span>
              <span class="measure-val" style="color: var(--accent-green);">98.4%</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  attachViewerEvents() {
    // Initialize Three.js instance
    setTimeout(() => {
      this.viewerInstance = new Drone3DViewer('viewer-container', {
        autoRotate: true,
      });

      // Mode toggles
      document.querySelectorAll('.viewer-btn-mode').forEach((btn) => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.viewer-btn-mode').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          const mode = btn.dataset.mode;
          if (this.viewerInstance) this.viewerInstance.setMode(mode);
        });
      });

      // Trajectory toggle
      const btnTraj = document.getElementById('btn-toggle-traj');
      if (btnTraj) {
        let trajVisible = true;
        btnTraj.addEventListener('click', () => {
          trajVisible = !trajVisible;
          btnTraj.classList.toggle('active', trajVisible);
          if (this.viewerInstance) this.viewerInstance.toggleTrajectory(trajVisible);
        });
      }

      // Auto rotate toggle
      const btnRotate = document.getElementById('btn-toggle-rotate');
      if (btnRotate) {
        let rotating = true;
        btnRotate.addEventListener('click', () => {
          rotating = !rotating;
          btnRotate.classList.toggle('active', rotating);
          if (this.viewerInstance) this.viewerInstance.toggleAutoRotate(rotating);
        });
      }

      // Reset camera
      const btnReset = document.getElementById('btn-reset-cam');
      if (btnReset) {
        btnReset.addEventListener('click', () => {
          if (this.viewerInstance) this.viewerInstance.resetCamera();
        });
      }
    }, 50);
  }

  /* ========================================================================
     6. Exports View
     ======================================================================== */
  renderExports() {
    return `
      <div class="view-header">
        <div>
          <h1 class="view-headline">Deliverables & Geospatial Exports</h1>
          <p class="view-subtitle">Download georeferenced 3D models, point clouds, and Cesium CZML flight paths.</p>
        </div>
      </div>

      <div class="projects-grid">
        <div class="project-card" style="padding: 24px;">
          <h3 class="project-card-title">OGC 3D Tiles 1.1 Archive</h3>
          <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 16px;">Hierarchical tileset for web-streaming in CesiumJS, Unreal Engine, and QGIS.</p>
          <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 20px;">
            Size: 184 MB • Format: .b3dm / tileset.json • CRS: EPSG:4978
          </div>
          <button class="btn-primary-action btn-dl" data-name="3d_tiles.zip" style="width: 100%; justify-content: center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            <span>Download 3D Tiles</span>
          </button>
        </div>

        <div class="project-card" style="padding: 24px;">
          <h3 class="project-card-title">Georeferenced LAS 1.4 Point Cloud</h3>
          <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 16px;">High-density calibrated points with RGB color and confidence classification flags.</p>
          <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 20px;">
            Size: 420 MB • Format: ASPRS LAS 1.4 • Points: 18.4M
          </div>
          <button class="btn-primary-action btn-dl" data-name="survey_points.las" style="width: 100%; justify-content: center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            <span>Download LAS Point Cloud</span>
          </button>
        </div>

        <div class="project-card" style="padding: 24px;">
          <h3 class="project-card-title">Cesium Trajectory (CZML & GeoJSON)</h3>
          <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 16px;">Generated directly from TASK-029: Electric Cyan flight path with orientation quaternions.</p>
          <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 20px;">
            Size: 2.8 MB • Format: .czml + .geojson • Waypoints: 420
          </div>
          <button class="btn-primary-action btn-dl" data-name="trajectory.czml" style="width: 100%; justify-content: center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            <span>Download CZML / GeoJSON</span>
          </button>
        </div>

        <div class="project-card" style="padding: 24px;">
          <h3 class="project-card-title">Textured 3D Mesh (GLB / OBJ)</h3>
          <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 16px;">Water-tight Poisson mesh with 8K texture atlas for CAD, Blender, and Unreal Engine.</p>
          <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 20px;">
            Size: 95 MB • Format: glTF binary (.glb) • Faces: 1.2M
          </div>
          <button class="btn-primary-action btn-dl" data-name="textured_model.glb" style="width: 100%; justify-content: center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
            <span>Download GLB Mesh</span>
          </button>
        </div>
      </div>
    `;
  }

  attachExportsEvents() {
    document.querySelectorAll('.btn-dl').forEach((btn) => {
      btn.addEventListener('click', () => {
        const filename = btn.dataset.name || 'export.bin';
        const dummyContent = JSON.stringify({
          platform: 'AEROMAP 3D',
          version: '2026.1',
          file: filename,
          generated_at: new Date().toISOString(),
          status: 'VERIFIED',
        }, null, 2);

        const blob = new Blob([dummyContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      });
    });
  }

  /* ========================================================================
     7. Settings View
     ======================================================================== */
  renderSettings() {
    return `
      <div style="max-width: 720px; margin: 0 auto; width: 100%;">
        <div class="view-header">
          <div>
            <h1 class="view-headline">Platform Settings</h1>
            <p class="view-subtitle">Configure coordinate reference systems, GPU cluster nodes, and S3 credentials.</p>
          </div>
        </div>

        <div class="quality-report-card">
          <h3 style="color: #FFFFFF; margin-bottom: 16px; font-family: var(--font-display);">Coordinate Reference System (CRS)</h3>
          <div style="margin-bottom: 20px;">
            <label style="display: block; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 6px;">Default Project Projection</label>
            <select style="width: 100%; padding: 10px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); color: #fff; border-radius: 6px;">
              <option>WGS 84 (EPSG:4326) - Global Lat/Lon</option>
              <option selected>UTM Zone 11N (EPSG:32611) - Metric Cartesian</option>
              <option>Web Mercator (EPSG:3857) - Mapping</option>
            </select>
          </div>

          <h3 style="color: #FFFFFF; margin-bottom: 16px; font-family: var(--font-display); padding-top: 16px; border-top: 1px solid var(--border-subtle);">Compute Node Cluster</h3>
          <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-cyan); margin-bottom: 8px;">
            Active GPU Worker: NVIDIA GeForce RTX 4090 • 24 GB VRAM
          </div>
          <div style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 20px;">
            Mixed-precision FP16 / BF16 active • PyTorch 2.3+ CUDA runtime connected.
          </div>

          <button class="btn-primary-action" style="padding: 10px 20px;">Save Settings</button>
        </div>
      </div>
    `;
  }
}

// Boot application
window.addEventListener('DOMContentLoaded', () => {
  window.__app = new AppController();
});
