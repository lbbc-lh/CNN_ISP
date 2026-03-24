# Debug UI Design

## Summary

Add a lightweight browser UI to the existing FastAPI project so the ISP debugging demo can be used without manual base64 conversion or raw API calls.

The UI is explicitly a debugging panel rather than a productized report page. It should prioritize dense, practical visibility into the current pipeline: uploaded source image, segmentation overlay, region masks, scores, region validity, dynamic weights, suggestions, and raw JSON.

## Goals

- Keep the existing `/analyze` API unchanged and continue to use it as the single analysis backend.
- Add a browser-accessible page at `/`.
- Let the user upload a local image file from the browser.
- Convert the file to base64 in the browser and call `/analyze`.
- Display:
  - source image
  - segmentation overlay
  - person/sky/vegetation/background masks
  - final score
  - global metrics
  - region metrics
  - effective weights
  - missing regions
  - suggestions
  - raw JSON response
- Keep the UI usable on both desktop and mobile.
- Keep implementation lightweight: FastAPI templates + static assets, no frontend build system.

## Non-Goals

- No React/Vite/SPA stack.
- No authentication or persistence.
- No multi-user session state.
- No image gallery/history page.
- No editing or annotation tools.

## Architecture

### Backend

Use the existing FastAPI server.

Add:

- a template route at `/`
- static file mounting for CSS/JS at `/static`
- `jinja2` runtime dependency for FastAPI template rendering

Do not duplicate analysis logic in the UI route. The page should fetch `/analyze` over HTTP from client-side JavaScript.

### Frontend

Add:

- `templates/index.html`
- `static/style.css`
- `static/app.js`

The frontend should:

1. accept local image selection
2. preview the source image immediately
3. convert the file to base64 with `FileReader`
4. POST to `/analyze`
5. render the returned data into debug panels

## Layout

Use a single debug dashboard page with these sections:

1. Header
   - project title
   - short description
   - upload input
   - analyze button
   - request status text

2. Main visual area
   - original image card
   - overlay image card

3. Mask area
   - person mask
   - sky mask
   - vegetation mask
   - background mask

4. Summary area
   - final score
   - suggestions
   - missing regions
   - effective weights

5. Metrics area
   - global metrics card
   - region metrics table or cards

6. Raw JSON area
   - collapsible or preformatted response dump

## Interaction Requirements

- The analyze button should stay disabled until a file is selected.
- While a request is in flight:
  - show loading state
  - disable repeated submit
- On success:
  - keep the selected source image visible
  - populate all visualization/metric sections
- On failure:
  - show a readable error message
  - keep the previous successful results visible if present

## Data Contract

The page consumes the current API response, including:

- `request_id`
- `global_metrics`
- `region_metrics`
- `final_score`
- `effective_weights`
- `score_breakdown`
- `missing_regions`
- `detected_regions`
- `suggestions`
- `visualizations.overlay_base64`
- `visualizations.mask_base64`

## Visual Direction

Use a tool-panel aesthetic:

- light background
- dark text
- restrained accent color
- high information density
- simple cards and tables
- no decorative animations beyond minimal loading feedback

The UI should feel like an internal engineering dashboard, not a marketing page.

## Testing

Add tests for:

- homepage route returns HTML
- static UI files exist if necessary for route coverage
- API tests remain green

JavaScript-heavy rendering does not need browser automation in this iteration.

## Implementation Readiness

This is ready for planning and implementation.
