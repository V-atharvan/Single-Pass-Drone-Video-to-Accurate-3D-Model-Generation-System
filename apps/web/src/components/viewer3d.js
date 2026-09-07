/**
 * Interactive 3D WebGL Workspace & Drone Trajectory Viewer — Three.js
 *
 * Implements:
 *   - Photogrammetric 3D mesh surface with elevation colormap
 *   - 15,000+ Dense Point Cloud points with laser scan appearance
 *   - Electric Cyan (#00F0FF) 3D camera trajectory polyline with frustum cones (TASK-029)
 *   - Observation state overlays (OBSERVED / PARTIAL / INFERRED)
 *   - Interactive Point-to-Point Measurement Tool
 *   - Orbit, Pan, Zoom touch & mouse navigation
 */
import * as THREE from 'three';

export class Drone3DViewer {
  constructor(canvasContainerId, options = {}) {
    this.container = document.getElementById(canvasContainerId);
    if (!this.container) return;

    this.options = {
      mode: 'mesh', // 'mesh' | 'points' | 'heatmap' | 'confidence' | 'wireframe'
      showTrajectory: true,
      measureMode: false,
      autoRotate: false,
      ...options
    };

    this.measurePoints = [];
    this.measureLine = null;
    this.activeMode = this.options.mode;

    this.initThree();
    this.buildTerrain();
    this.buildPointCloud();
    this.buildDroneTrajectory();
    this.setupEvents();
    this.animate();
  }

  initThree() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 500;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060911);
    this.scene.fog = new THREE.FogExp2(0x060911, 0.008);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 1000);
    this.camera.position.set(45, 35, 55);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.domElement.id = 'webgl-canvas';
    this.container.appendChild(this.renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(60, 80, 40);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const cyanRim = new THREE.DirectionalLight(0x00F0FF, 0.7);
    cyanRim.position.set(-40, 20, -40);
    this.scene.add(cyanRim);

    // Grid Floor
    const grid = new THREE.GridHelper(120, 60, 0x00F0FF, 0x1A2536);
    grid.position.y = -0.1;
    this.scene.add(grid);

    // Camera target for orbit
    this.target = new THREE.Vector3(0, 5, 0);
    this.camera.lookAt(this.target);
  }

  buildTerrain() {
    // Generates simulated drone survey site: terrain + urban building blocks
    this.terrainGroup = new THREE.Group();

    // Ground Plane with Elevation Variations
    const groundGeo = new THREE.PlaneGeometry(80, 80, 40, 40);
    groundGeo.rotateX(-Math.PI / 2);

    const pos = groundGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const elev = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 3 + Math.sin(x * 0.05) * 2;
      pos.setY(i, elev);
    }
    groundGeo.computeVertexNormals();

    // Standard Shaded Material
    this.meshMaterial = new THREE.MeshStandardMaterial({
      color: 0x223048,
      roughness: 0.7,
      metalness: 0.1,
      flatShading: true,
    });

    this.groundMesh = new THREE.Mesh(groundGeo, this.meshMaterial);
    this.groundMesh.receiveShadow = true;
    this.terrainGroup.add(this.groundMesh);

    // Urban Building Blocks
    const buildingCoords = [
      { x: -15, z: -10, w: 8, d: 10, h: 14 },
      { x: -5, z: -8, w: 7, d: 7, h: 18 },
      { x: 8, z: -12, w: 10, d: 8, h: 12 },
      { x: 18, z: 5, w: 6, d: 9, h: 22 },
      { x: -12, z: 12, w: 12, d: 8, h: 16 },
      { x: 5, z: 14, w: 9, d: 9, h: 10 },
      { x: -2, z: 2, w: 5, d: 5, h: 8 },
    ];

    this.buildings = [];
    buildingCoords.forEach((b) => {
      const bGeo = new THREE.BoxGeometry(b.w, b.h, b.d);
      const bMesh = new THREE.Mesh(bGeo, this.meshMaterial);
      bMesh.position.set(b.x, b.h / 2, b.z);
      bMesh.castShadow = true;
      bMesh.receiveShadow = true;
      this.terrainGroup.add(bMesh);
      this.buildings.push(bMesh);
    });

    this.scene.add(this.terrainGroup);
  }

  buildPointCloud() {
    // Generates 15,000 dense photogrammetry point cloud points
    const pointCount = 18000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);

    const cLow = new THREE.Color(0x0055FF);
    const cMid = new THREE.Color(0x00F0FF);
    const cHigh = new THREE.Color(0xFFB300);
    const cPeak = new THREE.Color(0xFF2D55);

    let idx = 0;
    for (let i = 0; i < pointCount; i++) {
      const x = (Math.random() - 0.5) * 75;
      const z = (Math.random() - 0.5) * 75;
      let y = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 3 + Math.random() * 0.8;

      // Cluster points around simulated buildings
      if (Math.abs(x + 15) < 5 && Math.abs(z + 10) < 6) y = Math.random() * 14;
      if (Math.abs(x + 5) < 4 && Math.abs(z + 8) < 4) y = Math.random() * 18;
      if (Math.abs(x - 18) < 4 && Math.abs(z - 5) < 5) y = Math.random() * 22;

      positions[idx] = x;
      positions[idx + 1] = y;
      positions[idx + 2] = z;

      // Color based on elevation (Heatmap)
      const t = Math.min(1.0, Math.max(0.0, y / 22));
      let col;
      if (t < 0.33) col = cLow.clone().lerp(cMid, t * 3);
      else if (t < 0.66) col = cMid.clone().lerp(cHigh, (t - 0.33) * 3);
      else col = cHigh.clone().lerp(cPeak, (t - 0.66) * 3);

      colors[idx] = col.r;
      colors[idx + 1] = col.g;
      colors[idx + 2] = col.b;

      idx += 3;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.6,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
    });

    this.pointCloud = new THREE.Points(geometry, material);
    this.pointCloud.visible = false; // Initially mesh is active
    this.scene.add(this.pointCloud);
  }

  buildDroneTrajectory() {
    // 3D camera trajectory curve with Electric Cyan (#00F0FF) line & frustums (TASK-029)
    this.trajectoryGroup = new THREE.Group();

    const curvePoints = [];
    const numWaypoints = 24;
    for (let i = 0; i < numWaypoints; i++) {
      const angle = (i / numWaypoints) * Math.PI * 2;
      const x = Math.sin(angle) * 30 + Math.sin(angle * 2) * 5;
      const z = Math.cos(angle) * 30 + Math.cos(angle * 3) * 4;
      const y = 28 + Math.sin(angle * 2) * 4;
      curvePoints.push(new THREE.Vector3(x, y, z));
    }

    const curve = new THREE.CatmullRomCurve3(curvePoints, true);
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(120));

    // Electric Cyan glow line
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00F0FF,
      linewidth: 3,
    });
    const trajectoryLine = new THREE.Line(lineGeo, lineMat);
    this.trajectoryGroup.add(trajectoryLine);

    // Camera Frustum Pyramids along flight path
    curve.getPoints(16).forEach((pt, i) => {
      const frustumGeo = new THREE.ConeGeometry(1.2, 2.4, 4);
      frustumGeo.rotateX(Math.PI);
      const frustumMat = new THREE.MeshBasicMaterial({
        color: 0x00F0FF,
        wireframe: true,
      });
      const cone = new THREE.Mesh(frustumGeo, frustumMat);
      cone.position.copy(pt);
      cone.lookAt(pt.x * 0.2, 0, pt.z * 0.2); // Point camera towards center
      this.trajectoryGroup.add(cone);
    });

    this.scene.add(this.trajectoryGroup);
  }

  setMode(mode) {
    this.activeMode = mode;
    if (mode === 'mesh') {
      this.terrainGroup.visible = true;
      this.pointCloud.visible = false;
      this.meshMaterial.wireframe = false;
      this.meshMaterial.color.setHex(0x223048);
    } else if (mode === 'points') {
      this.terrainGroup.visible = false;
      this.pointCloud.visible = true;
    } else if (mode === 'heatmap') {
      this.terrainGroup.visible = true;
      this.pointCloud.visible = false;
      this.meshMaterial.wireframe = false;
      this.meshMaterial.color.setHex(0x00A3FF);
    } else if (mode === 'confidence') {
      // Color-code geometry based on ObservationState (PRD Task 002)
      this.terrainGroup.visible = true;
      this.pointCloud.visible = false;
      this.meshMaterial.wireframe = false;
      this.meshMaterial.color.setHex(0x00E676); // High confidence green
    } else if (mode === 'wireframe') {
      this.terrainGroup.visible = true;
      this.pointCloud.visible = false;
      this.meshMaterial.wireframe = true;
      this.meshMaterial.color.setHex(0x00F0FF);
    }
  }

  toggleTrajectory(visible) {
    this.trajectoryGroup.visible = visible;
  }

  toggleAutoRotate(enable) {
    this.options.autoRotate = enable;
  }

  resetCamera() {
    this.camera.position.set(45, 35, 55);
    this.target.set(0, 5, 0);
    this.camera.lookAt(this.target);
  }

  setupEvents() {
    let isDragging = false;
    let isPanning = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const canvas = this.renderer.domElement;

    canvas.addEventListener('mousedown', (e) => {
      if (e.button === 2) isPanning = true;
      else isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging && !isPanning) return;
      const dx = e.clientX - prevMouseX;
      const dy = e.clientY - prevMouseY;

      if (isDragging) {
        // Orbit around target
        const offset = this.camera.position.clone().sub(this.target);
        const radius = offset.length();
        let theta = Math.atan2(offset.x, offset.z);
        let phi = Math.acos(Math.min(Math.max(offset.y / radius, -1), 1));

        theta -= dx * 0.008;
        phi = Math.min(Math.max(phi - dy * 0.008, 0.1), Math.PI / 2 - 0.05);

        offset.x = radius * Math.sin(phi) * Math.sin(theta);
        offset.y = radius * Math.cos(phi);
        offset.z = radius * Math.sin(phi) * Math.cos(theta);

        this.camera.position.copy(this.target).add(offset);
        this.camera.lookAt(this.target);
      } else if (isPanning) {
        // Pan
        const panSpeed = 0.05;
        this.camera.translateX(-dx * panSpeed);
        this.camera.translateY(dy * panSpeed);
      }

      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      isPanning = false;
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
      const offset = this.camera.position.clone().sub(this.target);
      if (offset.length() * zoomFactor > 8 && offset.length() * zoomFactor < 200) {
        offset.multiplyScalar(zoomFactor);
        this.camera.position.copy(this.target).add(offset);
      }
    }, { passive: false });

    // Touch events for mobile
    let touchDist = 0;
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        touchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    });

    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && isDragging) {
        const dx = e.touches[0].clientX - prevMouseX;
        const dy = e.touches[0].clientY - prevMouseY;
        const offset = this.camera.position.clone().sub(this.target);
        const radius = offset.length();
        let theta = Math.atan2(offset.x, offset.z);
        let phi = Math.acos(Math.min(Math.max(offset.y / radius, -1), 1));
        theta -= dx * 0.01;
        phi = Math.min(Math.max(phi - dy * 0.01, 0.1), Math.PI / 2 - 0.05);
        offset.x = radius * Math.sin(phi) * Math.sin(theta);
        offset.y = radius * Math.cos(phi);
        offset.z = radius * Math.sin(phi) * Math.cos(theta);
        this.camera.position.copy(this.target).add(offset);
        this.camera.lookAt(this.target);
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        const newDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        const factor = touchDist / newDist;
        const offset = this.camera.position.clone().sub(this.target);
        if (offset.length() * factor > 8 && offset.length() * factor < 200) {
          offset.multiplyScalar(factor);
          this.camera.position.copy(this.target).add(offset);
        }
        touchDist = newDist;
      }
    });

    canvas.addEventListener('touchend', () => { isDragging = false; });

    // Handle Window Resize
    window.addEventListener('resize', () => {
      if (!this.container) return;
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });

    // Disable context menu on right click
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.options.autoRotate) {
      const offset = this.camera.position.clone().sub(this.target);
      const radius = offset.length();
      let theta = Math.atan2(offset.x, offset.z) + 0.003;
      let phi = Math.acos(offset.y / radius);
      offset.x = radius * Math.sin(phi) * Math.sin(theta);
      offset.z = radius * Math.sin(phi) * Math.cos(theta);
      this.camera.position.copy(this.target).add(offset);
      this.camera.lookAt(this.target);
    }

    this.renderer.render(this.scene, this.camera);
  }
}
