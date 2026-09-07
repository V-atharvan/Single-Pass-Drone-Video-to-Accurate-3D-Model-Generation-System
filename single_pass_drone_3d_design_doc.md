# Design Document
## Single-Pass Drone Video to Accurate 3D Model Generation System

**Version:** 1.0  
**Status:** Proposed  
**Design Direction:** Bubblegum Pop / Technical Geospatial  
**Primary Platforms:** Web application + GPU reconstruction platform

---

# 1. Design Overview

The product is an AI-powered platform that transforms a **single drone video pass** into an interactive, georeferenced 3D representation of the captured scene.

The design combines two seemingly different qualities:

1. **Technical precision** required for mapping, inspection, measurement, and reconstruction.
2. **A bold, approachable visual language** inspired by the supplied design reference.

The interface should therefore feel:

- Minimal
- Bright
- Geometric
- Spatial
- Modern
- Confident
- Highly visual

The design should avoid the typical dark military/GIS dashboard aesthetic unless the user explicitly chooses a dark operational mode.

---

# 2. Design Language

## 2.1 Visual Concept

The supplied reference establishes a visual language based on:

```text
Bubblegum Pink   #FF69B4
Deep Teal        #069494
White            #FFFFFF
Electric Cyan    #00F0FF
```

The product should translate these colors into a technical geospatial interface.

### Design principle

> **Playful surface, serious information.**

The reconstruction engine can be highly sophisticated without forcing the interface to look like an aircraft cockpit from a 1990s science-fiction movie.

---

# 3. Color System

## 3.1 Primary Palette

| Token | Color | Hex | Usage |
|---|---|---|---|
| Primary Pink | Bubblegum Pink | `#FF69B4` | Primary actions, highlights, selected states |
| Primary Teal | Deep Teal | `#069494` | Navigation, secondary actions, data panels |
| Background | White | `#FFFFFF` | Main application background |
| Accent Cyan | Electric Cyan | `#00F0FF` | 3D overlays, active visualization, telemetry |

---

## 3.2 Supporting Colors

Use a restrained neutral system around the supplied palette.

```text
Ink
#111111

Muted
#6B6B6B

Soft Gray
#F4F4F4

Border
#E5E5E5

Success
#20A36A

Warning
#F0A000

Error
#D64545
```

The four primary brand colors should remain visually dominant.

---

# 4. Color Usage Rules

## Pink

Use for:

- Primary CTA
- Selected project
- Active navigation
- Important actions
- Key highlights
- Progress emphasis

Example:

```text
[ Start Reconstruction ]
```

should use the pink brand treatment.

## Teal

Use for:

- Secondary controls
- Metadata panels
- Navigation sections
- Spatial information
- Technical status

## Cyan

Use for:

- Camera trajectory
- Active 3D overlays
- Selection outlines
- Measurement guides
- Depth/confidence visualization

## White

Use as:

- Main canvas
- Cards
- Empty space
- Clean visual separation

---

# 5. Typography

## Primary Typeface

Recommended:

**Inter**

Alternative:

**Geist**

The typography should be clean, geometric, and highly legible.

---

## Type Scale

```text
Display
64px / 1.0

Page Heading
40px / 1.1

Section Heading
28px / 1.2

Card Heading
20px / 1.3

Body
16px / 1.5

Small
14px / 1.4

Metadata
12px / 1.4
```

Use large typography sparingly.

The landing page can be expressive. The operational dashboard should prioritize information density.

---

# 6. Typography Rules

### Headings

Use:

- Large sizes
- Tight letter spacing
- Medium/regular weight
- Short phrases

Example:

```text
Turn one flight
into a 3D world.
```

### Body

Use neutral, highly readable text.

### Metrics

Use large numerals.

Example:

```text
93%
Coverage
```

or:

```text
0.32 m
Horizontal RMSE
```

---

# 7. Layout System

The application should use a modular grid.

## Desktop

```text
┌──────────────┬─────────────────────────────┐
│              │                             │
│   Sidebar    │       Main Workspace        │
│              │                             │
│              │                             │
└──────────────┴─────────────────────────────┘
```

Recommended maximum content width: `1440px`.

## Mobile

```text
┌─────────────────────────────┐
│          Header             │
├─────────────────────────────┤
│                             │
│       Main Content          │
│                             │
│                             │
├─────────────────────────────┤
│ Dashboard Projects Upload   │
│  3D View       More         │
└─────────────────────────────┘
```

The mobile bottom navigation must not be treated as an optional enhancement. It is a core responsive navigation requirement.

---

# 8. Application Shell

## Desktop Shell
On desktop and large tablet screens (>= 1024px):
- Left sidebar or top navigation depending on page context.
- Provides immediate access to:
  - Dashboard
  - Projects
  - Upload
  - Processing
  - 3D Workspace
  - Exports
  - Settings
- Status / Telemetry / Job Progress docked at bottom or inline.

## Mobile Shell
On mobile screens (< 768px):
- Minimalist top header containing brand emblem, active job pulse badge, and user avatar.
- Fixed bottom navigation bar remaining accessible during main application navigation:
  `[ Dashboard ] [ Projects ] [ Upload ] [ 3D View ] [ More ]`
- Requirements:
  - Fixed to the bottom of the viewport (`position: fixed; bottom: 0;`).
  - Safe-area aware for modern mobile devices (`padding-bottom: calc(12px + env(safe-area-inset-bottom))`).
  - Does not overlap important content (main content container applies matching bottom padding).
  - Touch-friendly navigation targets (minimum 44 x 44 px).
  - Clear active-state indicator (brand pink `#FF69B4` underline or pill background).
  - Icon + text label where space permits.
  - Horizontally balanced across screen width.
  - Accessible with keyboard and screen readers (`role="navigation"`, `aria-label="Mobile Navigation"`).
  - Navigation must remain usable in both portrait and landscape modes.

---

# 9. Dashboard Design

The dashboard should answer three questions immediately:

1. What projects exist?
2. What is currently processing?
3. What models are ready?

### Layout

```text
┌──────────────────────────────────────────────┐
│ Projects                          + New      │
├──────────────────────────────────────────────┤
│                                              │
│  Project Card       Project Card             │
│                                              │
│  3D Preview         3D Preview               │
│  12 models          4 flights                │
│                                              │
├──────────────────────────────────────────────┤
│ Active Reconstruction Jobs                   │
│                                              │
│ Flight 032   ███████████░░░  78%             │
│ Flight 033   █████░░░░░░░░  42%              │
└──────────────────────────────────────────────┘
```

---

# 10. Project Card

Each project card should contain:

```text
Project Name
Location
Last Updated
Flight Count
Model Count
Quality Score
```

Visual preview:

- 3D thumbnail
- Aerial image
- Simplified terrain rendering

The card should have a large visual area and minimal text.

---

# 11. Upload Experience

The upload experience should be one of the simplest parts of the product.

## Upload Screen

```text
┌─────────────────────────────────────────────┐
│                                             │
│            Drop drone video here            │
│                                             │
│       MP4 / MOV · 1080p / 4K                │
│                                             │
│               [ Browse files ]              │
│                                             │
└─────────────────────────────────────────────┘

GPS Data
[ Upload GPS ]

Flight Metadata
[ Automatically detected ]
```

Use the pink accent for the upload action.

---

# 12. Input Quality Screen

Before reconstruction, show the user whether the dataset is usable.

```text
Flight Quality

Video Quality       Good       91%
GPS Quality         Moderate   74%
Motion Blur         Low        88%
Scene Texture       Good       94%
Camera Metadata     Complete   100%

Expected Result
HIGH
```

Use clear visual indicators rather than technical jargon alone.

---

# 13. Reconstruction Progress

The processing screen should feel alive without becoming distracting.

### Main view

```text
RECONSTRUCTING

Building your 3D scene

██████████████████░░░░░░  78%

Current stage
Depth estimation

Frames
8,421 / 11,672

Estimated time
03:12
```

### Processing stages

```text
✓ Input validation
✓ Frame extraction
✓ Pose estimation
● Depth estimation
○ Scene fusion
○ Mesh generation
○ Texturing
○ Quality assessment
```

---

# 14. 3D Viewer

The 3D viewer is the primary product surface.

## Layout

```text
┌──────────────────────────────────────────────────────┐
│ Project / Scene                       Export         │
├──────────────────────────────────────────────────────┤
│                                                      │
│                                                      │
│                   3D SCENE                           │
│                                                      │
│                                                      │
│                         +                            │
│                       camera                         │
│                         path                         │
│                                                      │
├───────────────┬──────────────────────────────────────┤
│ Layers        │ Measurement / Information            │
│               │                                      │
│ ☑ Terrain     │ Select a point                       │
│ ☑ Buildings   │                                      │
│ ☑ Roads       │ Distance: --                         │
│ ☑ Vegetation  │ Height: --                           │
│ ☑ Point Cloud │                                      │
│ ☑ Confidence  │ Coordinates: --                      │
└───────────────┴──────────────────────────────────────┘
```

---

# 15. 3D Visualization Language

The 3D environment should use:

- White/neutral scene background
- Cyan active geometry
- Pink selections
- Teal data overlays

### Camera trajectory

Use cyan.

### Selected object

Use pink.

### Confidence visualization

Suggested gradient:

```text
High confidence
    ↓
Cyan / Teal

Medium confidence
    ↓
Neutral

Low confidence
    ↓
Pink
```

The exact visualization should remain accessible to users with color-vision differences by also using opacity, patterns, or labels.

---

# 16. Layer Control

Users should be able to toggle:

```text
Terrain
Buildings
Roads
Infrastructure
Vegetation
Vehicles
Point Cloud
Mesh
Textures
Camera Path
Confidence
AI Inferred Areas
```

Each layer should have:

- Visibility toggle
- Opacity
- Optional color treatment

---

# 17. Measurement UI

Measurement tools should appear contextually.

### Distance

```text
DISTANCE

12.48 m

Estimated error
±0.18 m
```

### Height

```text
STRUCTURE HEIGHT

18.2 m

Estimated error
±0.31 m
```

### Area

```text
AREA

2,481 m²
```

Do not hide uncertainty.

A number without its confidence is often just typography pretending to be science.

---

# 18. Model Quality Panel

The quality panel should make accuracy understandable.

```text
MODEL QUALITY
────────────────────

Overall
87 / 100

Coverage
93%

Horizontal RMSE
0.32 m

Vertical RMSE
0.58 m

Geometry
High

Geolocation
Medium

Texture
High
```

---

# 19. Confidence Visualization

The model should support a dedicated confidence mode.

Example:

```text
Observed
████████████████ 84%

Partially Observed
████ 10%

AI Inferred
██ 6%
```

Users should be able to click an area and inspect why its confidence is high or low.

---

# 20. Semantic Object Panel

Clicking an object should open a contextual panel.

Example:

```text
BUILDING

Type
Commercial

Height
18.2 m

Footprint
1,420 m²

Geometry confidence
94%

Texture confidence
87%

Observation
Mostly observed
```

---

# 21. Export UI

Export should be straightforward.

```text
EXPORT MODEL

3D Model
○ GLB
○ OBJ
○ glTF

Point Cloud
○ LAS
○ LAZ
○ PLY

Terrain
○ GeoTIFF

Streaming
○ 3D Tiles

[ Export ]
```

Advanced settings should remain hidden unless needed.

---

# 22. Component Design

## Buttons

### Primary

```text
Pink background
Black/white text depending on contrast
Rounded corners
```

### Secondary

```text
White background
Teal border
Teal text
```

### Tertiary

```text
Text only
```

---

# 23. Cards

Cards should use:

```text
White background
Subtle border
Minimal radius
Large internal spacing
```

Avoid excessive shadows.

Suggested radius:

```text
8px – 16px
```

---

# 24. Status Components

Use concise labels:

```text
READY
PROCESSING
WARNING
FAILED
COMPLETED
```

Avoid ambiguous statuses such as:

```text
Almost done
Working...
Something happened
```

---

# 25. Progress Indicators

Use the brand colors sparingly.

Example:

```text
████████████████░░░░
78%
```

The progress bar should be visually strong but not dominate the page.

---

# 26. Iconography

Use a consistent geometric icon set.

Recommended:

**Lucide Icons**

Use icons for:

- Upload
- Flight
- Map
- Layers
- Measurement
- Settings
- Export
- Search
- Camera
- Terrain
- Building
- Warning

Icons should support labels rather than replace them.

---

# 27. Navigation

## Desktop Navigation (Sidebar / Top Bar)
Primary navigation items:
- Dashboard
- Projects
- Upload
- Processing
- 3D Workspace
- Exports
- Settings

## Mobile Navigation (Fixed Bottom Bar)
Primary bottom navigation items:
- Dashboard (`/dashboard`)
- Projects (`/projects`)
- Upload (`/upload`)
- 3D View (`/workspace/3d`)
- More (Opens swipeable bottom sheet drawer with: Processing status, Exports, Settings, Team, API documentation, Logout)

Keep the navigation shallow, fast, and accessible.

---

# 28. Landing Page

The landing page should be dramatically simpler than the application.

## Hero

```text
ONE FLIGHT.
ONE VIDEO.
ONE 3D WORLD.

Turn a single drone pass
into a measurable 3D scene.

[ Start Reconstruction ]

[ Explore Demo ]
```

Visual:

- Large 3D terrain scene
- Cyan camera trajectory
- Pink highlighted building
- Teal terrain
- Minimal interface chrome

---

# 29. Landing Page Structure

```text
Hero
 ↓
How It Works
 ↓
3D Reconstruction Demo
 ↓
Accuracy
 ↓
Use Cases
 ↓
Exports
 ↓
Security
 ↓
CTA
```

---

# 30. How It Works Section

Use a four-step visual flow:

```text
01
CAPTURE

Single drone pass

↓

02
PROCESS

AI estimates pose,
depth and scene structure

↓

03
RECONSTRUCT

Fuse geometry,
textures and geospatial data

↓

04
ANALYZE

Measure, inspect,
export
```

---

# 31. Use Case Design

Use large visual cards.

```text
DISASTER RESPONSE

Rapid 3D situational awareness
after a single flight.
```

```text
INFRASTRUCTURE

Inspect structures without
multiple flight passes.
```

```text
CONSTRUCTION

Track site geometry
and progress.
```

```text
MAPPING

Create georeferenced
3D scene data.
```

---

# 32. Responsive Design

The platform is a fully responsive web application, not a desktop-only dashboard. Every major workflow must remain fully usable across devices:
`Login → Dashboard → Create Project → Upload Drone Video → Quality Report → Processing Status → 3D Workspace → Measurement → Analysis → Export`

Desktop and mobile must feel like the same unified product, rather than two unrelated interfaces squeezed into the same CSS file.

## 32.1 Breakpoint Behavior

- **Mobile (< 768px)**: Compact single-column layouts, fixed bottom navigation bar, full-screen modals or swipeable bottom sheets, touch-optimized measurement tools.
- **Tablet (768px — 1023px)**: Adaptive 2-column grids, collapsible side drawers, touch and pointer dual support.
- **Desktop (1024px — 1439px)**: Full sidebar navigation, persistent 3D inspector panels, split-screen video/3D comparison views.
- **Large Desktop (>= 1440px)**: Expanded workspace, multi-panel analytics, maximum content width capped at 1440px with generous whitespace.

Components must reflow rather than simply shrink. Tables, cards, forms, dialogs, charts, processing timelines, and 3D controls must have mobile-specific layouts where necessary.

## 32.2 Component Reflow Rules

- **Tables**: Reflow into touch-friendly cards on mobile, showing key identifiers and status badges, with tap-to-expand details.
- **Metric Grids**: Reflow from 4-column desktop grids to 2-column or single-column vertical stacks.
- **Processing Timeline**: Reflows from horizontal stepper to vertical chronological timeline with animated status pulses.
- **Dialogs & Modals**: Desktop centered modals become full-screen overlays or bottom sheet drawers on mobile.
- **Form Controls**: Stack vertically on mobile with full-width inputs and minimum 44px tap heights.

## 32.3 Mobile Upload Experience

The upload workflow must be completely mobile-responsive:
- Select drone video directly from device camera roll or file manager.
- Upload large video files (multi-GB) using chunked direct-to-S3 uploads.
- View real-time upload progress (MB/s speed, % completed, estimated remaining time).
- Pause and resume support where technically supported.
- Clear error communication with retry actions.
- Automatically advance to pre-flight quality report and processing status after upload.
- Upload progress offloaded to Web Workers / background threads so the UI never freezes or stutters.

## 32.4 Mobile 3D Workspace

The 3D viewer must seamlessly adapt to mobile touchscreens:
- **Touch Navigation**:
  - One-finger drag: Rotate / tilt camera.
  - Two-finger pinch: Zoom in / out.
  - Two-finger drag: Pan across terrain.
- **Fullscreen Mode**: One-tap toggle to hide browser chrome and UI overlays for maximum viewing area.
- **Bottom Sheets & Drawers**: Complex desktop controls collapse into contextual swipeable bottom sheets:
  - Layer visibility (Mesh, Point Cloud, Terrain, Camera Path).
  - Observation state filters (all 6 states: Observed, Partial, Inferred, Unknown, Dynamic Excluded, Low Confidence).
  - Semantic object inspector details.
- **Mobile Measurement**: Tap-to-place pin points with precision drag loupe; floating mobile HUD displays measured distance, height, error margin, and coordinates without obscuring the 3D scene.

---

# 33. Accessibility

Requirements:

- WCAG 2.2 AA target compliance across mobile and desktop.
- Touch target sizes: Minimum 44 x 44 px for all interactive elements.
- Keyboard navigation with visible, high-contrast focus rings.
- Screen-reader labels on all icon-only buttons and navigation landmarks (`aria-label`, `aria-current`).
- Safe-area awareness: All mobile bars and sticky headers respect `env(safe-area-inset-top)` and `env(safe-area-inset-bottom)`.
- Sufficient contrast ratios: Brand colors tested against WCAG contrast algorithms; never rely on color alone.
- Reduced-motion support: Respect `prefers-reduced-motion` for all UI animations and 3D camera transitions.
- Semantic navigation landmarks (`<header>`, `<nav>`, `<main>`, `<aside>`).
- Orientation support: Seamless usability in both portrait and landscape modes.

---

# 34. Interaction Principles

## 34.1 Progressive Disclosure

Show simple controls first.

Advanced controls:

```text
Camera
Coordinate System
Processing
Accuracy
Texture
Export
```

should be expandable.

## 34.2 Immediate Feedback

Every long-running operation should show:

- Current stage
- Progress
- Errors
- Estimated completion

## 34.3 Spatial Context

Whenever possible, information should connect to the 3D scene.

Example:

Click building → highlight building → show metrics.

---

# 35. Motion Design

Animations should be subtle and purposeful.

Use:

- Fade
- Scale
- Slide
- Progress transitions
- Camera transitions

Avoid:

- Excessive bouncing
- Continuous decorative motion
- Distracting 3D animations

Recommended durations:

```text
Micro interaction: 120–180 ms
Panel transition: 180–240 ms
Major transition: 300–450 ms
```

---

# 36. Loading States

Use meaningful skeletons.

Example:

```text
Loading project
████████████████░░░░
Preparing 3D scene...
```

For the 3D viewer:

```text
Loading terrain...
Loading buildings...
Loading textures...
```

---

# 37. Empty States

Empty states should teach the user what to do.

Example:

```text
NO FLIGHTS YET

Upload your first drone flight
to create a 3D scene.

[ Upload Flight ]
```

Avoid empty states that merely say:

```text
No data.
```

---

# 38. Error States

Example:

```text
RECONSTRUCTION PAUSED

The video contains too much
motion blur for reliable pose estimation.

Try:
• A higher-quality recording
• A slower flight
• A flight with less camera movement

[ View Input Report ]
```

The system should explain the problem and its consequences.

---

# 39. Design Tokens

Suggested token structure:

```css
:root {
  --color-pink: #FF69B4;
  --color-teal: #069494;
  --color-cyan: #00F0FF;
  --color-white: #FFFFFF;

  --color-ink: #111111;
  --color-muted: #6B6B6B;
  --color-surface: #F4F4F4;
  --color-border: #E5E5E5;

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;
  --space-12: 48px;
  --space-16: 64px;
}
```

---

# 40. Frontend Information Architecture

```text
/
├── Landing
│
├── /login
│
├── /dashboard
│
├── /projects
│   └── /[projectId]
│       ├── Overview
│       ├── Flights
│       ├── Models
│       └── Analytics
│
├── /flights
│   └── /[flightId]
│       ├── Input
│       ├── Quality
│       └── Processing
│
├── /models
│   └── /[modelId]
│       ├── Viewer
│       ├── Measurements
│       ├── Accuracy
│       └── Export
│
└── /settings
```

---

# 41. Technical Design

## Frontend

```text
Next.js
TypeScript
Tailwind CSS
Radix UI
CesiumJS
```

## Backend

```text
FastAPI
Python
PostgreSQL
PostGIS
Redis
S3
SQS
```

## AI

```text
PyTorch
CUDA
OpenCV
Open3D
GDAL
PDAL
```

---

# 42. 3D Rendering Architecture

```text
                    API
                     │
                     ▼
                Model Metadata
                     │
                     ▼
                    S3
                     │
             3D Tiles / GLB
                     │
                     ▼
                  CesiumJS
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
     Terrain      Buildings     Point Cloud
        │            │             │
        └────────────┼─────────────┘
                     ▼
              Interactive Scene
```

---

# 43. Frontend State Management

Use server state for:

- Projects
- Flights
- Jobs
- Models
- Measurements

Use local state for:

- Viewer settings
- Selected layer
- Camera state
- Measurement mode
- UI panels

Recommended approach:

```text
TanStack Query
+
React state
```

Avoid putting the entire 3D scene into global application state.

---

# 44. API / Frontend Contract

Use generated TypeScript types from backend schemas.

Recommended:

```text
FastAPI
    ↓
OpenAPI
    ↓
Generated TypeScript client/types
    ↓
Next.js
```

This reduces mismatches between reconstruction jobs and UI state.

---

# 45. Design for Reconstruction States

The UI must understand the reconstruction pipeline.

```text
QUEUED
VALIDATING
EXTRACTING
POSE
DEPTH
SEGMENTATION
FUSION
MESH
TEXTURE
GEOREFERENCING
QUALITY
TILES
COMPLETED
FAILED
```

Each state should have:

- Label
- Progress
- Description
- Expected next stage
- Error state where relevant

---

# 46. Design for Uncertainty

This is a critical product requirement.

The interface should never present inferred geometry as equivalent to directly observed geometry.

Use visual indicators:

```text
Observed
Solid geometry

Partially Observed
Reduced opacity

AI Inferred
Pattern / outline / alternate treatment

Unknown
No geometry
```

A legend should always be available in confidence mode.

---

# 47. Geospatial UI

Display:

```text
Latitude
Longitude
Altitude
CRS
Scale
North indicator
```

Optional:

```text
Grid
Coordinates
Compass
Mini-map
```

The UI should make the geographic context obvious without turning the entire interface into a GIS textbook.

---

# 48. Performance Requirements

## Viewer

Target:

```text
≥ 30 FPS
```

for normal operational scenes.

Use:

- 3D Tiles
- Level of Detail
- Frustum culling
- Progressive loading
- GPU rendering

## Dashboard

Target:

```text
First meaningful render < 2.5 seconds
```

under normal network conditions.

---

# 49. Design Performance Budget

Avoid:

- Huge uncompressed textures
- Loading entire point clouds immediately
- Rendering all model levels simultaneously
- Excessive client-side state
- Unnecessary animation

Use:

```text
Lazy loading
LOD
Progressive rendering
Asset compression
CDN caching
```

---

# 50. Security UX

Security should be visible but not intrusive.

Project settings may show:

```text
PRIVATE PROJECT

Only members of this organization
can access this scene.
```

Export actions should display:

```text
Export contains project geospatial data.
```

Sensitive operational deployments should support stronger organization and deployment controls.

---

# 51. Design System Components

Minimum component library:

```text
Button
IconButton
Input
Select
Dropdown
Modal
Drawer
Tabs
Badge
Status
Progress
Card
MetricCard
DataTable
UploadZone
FileCard
ProjectCard
FlightCard
JobCard
QualityScore
ConfidenceBadge
LayerControl
MeasurementTool
ViewerToolbar
ViewerPanel
MapMiniature
ExportDialog
Toast
Tooltip
```

---

# 52. Component Naming

Use semantic names.

Good:

```text
ReconstructionProgress
ModelQualityCard
MeasurementPanel
ConfidenceLegend
```

Avoid:

```text
BlueBox
BigPanel
ThingCard
MapStuff
```

Design systems should describe what something does, not what it looked like on Tuesday.

---

# 53. UX Metrics

Measure:

### Upload

- Upload completion rate
- Upload failure rate
- Time to first reconstruction

### Reconstruction

- Job start rate
- Job completion rate
- Processing abandonment

### Viewer

- Viewer load time
- Model interaction rate
- Measurement usage
- Layer usage

### Export

- Export rate
- Export failure rate
- Most-used formats

---

# 54. Design Success Criteria

The design is successful when a new user can:

1. Understand the product in under 10 seconds.
2. Create a project without training.
3. Upload a drone flight without confusion.
4. Understand whether the input is good enough.
5. Understand reconstruction progress.
6. Open the resulting 3D model immediately.
7. Identify observed versus inferred geometry.
8. Measure a structure.
9. Understand the reported accuracy.
10. Export the result.

---

# 55. Example Primary Screen

```text
┌─────────────────────────────────────────────────────────────┐
│  3D//ONE       Projects   Flights   Models       Account    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Riverside Survey                              READY        │
│  2026-09-08 · Flight 032                                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │                                                       │  │
│  │                    3D SCENE                           │  │
│  │                                                       │  │
│  │             ╱──────────────╲                          │  │
│  │            ╱    BUILDING    ╲                         │  │
│  │           ╱──────────────────╲                        │  │
│  │                                                       │  │
│  │     ───────── cyan camera trajectory ─────────         │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Coverage          Horizontal RMSE       Confidence         │
│    93%                 0.32 m               87%             │
│                                                             │
│  Layers       Measurements       Accuracy       Export      │
└─────────────────────────────────────────────────────────────┘
```

---

# 56. Design Direction Summary

The final product should feel like:

```text
Modern geospatial software
        +
AI laboratory
        +
Minimal design system
        +
Bold color
        +
Interactive 3D environment
```

It should **not** feel like:

```text
Traditional GIS
+
Military command dashboard
+
Spreadsheet-heavy enterprise software
```

The supplied Bubblegum Pop aesthetic provides the visual identity, while the underlying interaction model remains precise and professional.

---

# 57. Final Design Principles

### 01. Make the 3D scene the hero

The model is the product.

### 02. Use color with purpose

Pink, teal, cyan, and white should create hierarchy rather than decoration.

### 03. Keep controls minimal

Expose complexity progressively.

### 04. Make uncertainty visible

Observed and inferred geometry must never look identical.

### 05. Make measurements trustworthy

Every important measurement should provide context and uncertainty.

### 06. Preserve visual calm

Even when processing millions of points, the interface should remain understandable.

### 07. Design for operational speed

The user should move from:

```text
Upload
→ Process
→ Inspect
→ Measure
→ Export
```

with minimal friction.

---

# 58. Final Design Statement

> **Single-Pass 3D should make advanced aerial reconstruction feel simple: upload one flight, watch the scene emerge, inspect the geometry, understand the confidence, and use the resulting model.**

The visual system uses **Bubblegum Pink `#FF69B4`, Deep Teal `#069494`, White `#FFFFFF`, and Electric Cyan `#00F0FF`** as its defining palette, combined with clean typography, geometric components, large spatial canvases, and a 3D-first interaction model.

The result is a product that looks approachable while communicating the precision expected from professional geospatial software.
