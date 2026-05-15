# Design System Reference Guide

## Quick Start

The enhanced design system provides semantic CSS classes for all common UI patterns. All styles are defined in `globals.css` and use CSS custom properties for theming.

## Color Variables

```css
:root {
  --bg-primary: #0a0e27;      /* Dark navy background */
  --bg-secondary: #1a1f4d;    /* Secondary background */
  --accent: #00d4ff;          /* Primary cyan */
  --accent-alt: #ff006e;      /* Secondary magenta */
  --success: #00ff41;         /* Success green */
  --warning: #ffaa00;         /* Warning orange */
  --danger: #ff0055;          /* Danger red */
  
  --card-bg: rgba(26, 31, 77, 0.4);
  --card-border: rgba(0, 212, 255, 0.2);
  --card-border-hover: rgba(0, 212, 255, 0.4);
  --text-primary: #f0f9ff;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
}
```

## Component Classes

### Cards & Containers

#### `.stat-card`
Displays a statistic with icon, label, value, and trend.

```html
<div class="stat-card">
  <div class="stat-icon">📊</div>
  <div class="stat-content">
    <p class="stat-label">Models Scanned</p>
    <p class="stat-value">24</p>
    <p class="stat-trend">+3 this week</p>
  </div>
</div>
```

**Usage**: Dashboard overview cards, KPI displays
**Variants**: `.danger`, `.success` (add color styling)

---

#### `.stats-grid`
Responsive grid layout for stat cards.

```html
<div class="stats-grid">
  <div class="stat-card">...</div>
  <div class="stat-card">...</div>
  <!-- Auto-fits with 240px minimum width -->
</div>
```

**Breakpoints**:
- Desktop (1200px+): 4 columns
- Tablet (1024px-1200px): 2 columns  
- Mobile (<768px): 1 column

---

#### `.scan-section` & `.scan-info`
Two-column layout for form and sidebar.

```html
<div class="scan-layout">
  <div class="scan-section">
    <!-- Form content -->
  </div>
  <div class="scan-info">
    <!-- Info sidebar (sticky on desktop) -->
  </div>
</div>
```

**Features**: 
- `.scan-section`: Main content area
- `.scan-info`: Sticky sidebar (desktop only)
- Responsive: Stacks vertically below 1200px

---

### Forms

#### `.form-input`
Text input field with focus effects.

```html
<input type="text" class="form-input" placeholder="Enter value">
```

**Styling**:
- Transparent background with subtle border
- Cyan focus state with glow
- Proper padding and border radius

---

#### `.form-select`
Dropdown select with consistent styling.

```html
<select class="form-select">
  <option>Option 1</option>
  <option>Option 2</option>
</select>
```

**Features**:
- Matches `.form-input` styling
- Cyan focus state
- Dark background for options

---

#### `.form-row`
Two-column layout for form fields.

```html
<div class="form-row">
  <div class="form-group">
    <label>Field 1</label>
    <input class="form-input" />
  </div>
  <div class="form-group">
    <label>Field 2</label>
    <input class="form-input" />
  </div>
</div>
```

**Responsive**: Single column on mobile (<768px)

---

#### `.upload-trigger`
File upload area with drag-and-drop affordance.

```html
<div class="upload-trigger" onclick="selectFile()">
  <Upload size={28} />
  <p class="upload-title">Click to upload model</p>
  <p class="upload-subtitle">Supports .pt, .pth, .onnx</p>
</div>
```

**Styling**:
- Dashed border with cyan color
- Hover effect changes border and background
- Centered content with icon focus

---

#### `.radio-option` & `.radio-group`
Radio button group styling.

```html
<div class="radio-group">
  <label class="radio-option">
    <input type="radio" name="mode" />
    Upload File
  </label>
  <label class="radio-option">
    <input type="radio" name="mode" />
    Server Path
  </label>
</div>
```

**Features**:
- Consistent label styling
- Cyan accent color for selected state
- Proper spacing and alignment

---

### Buttons

#### `.scan-button`
Primary action button with gradient and effects.

```html
<button class="scan-button" onclick="startScan()">
  <Zap size={20} />
  Start Security Scan
</button>
```

**Styling**:
- Gradient background (cyan to teal)
- Glowing shadow effect
- Hover elevation (translateY)
- Disabled state with reduced opacity
- Loading state with spinner animation

**States**:
- Normal: Cyan gradient with glow
- Hover: Enhanced glow, slight lift
- Active: Slight depression
- Disabled: Faded, no interaction

---

#### `.btn-primary`, `.btn-secondary`, `.btn-danger`
Standard button variants (from existing system).

```html
<button class="btn-primary">Primary Action</button>
<button class="btn-secondary">Secondary Action</button>
<button class="btn-danger">Delete</button>
```

---

### Tables

#### `.scans-table`
Professional table styling with hover effects.

```html
<table class="scans-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Status</th>
      <th>Risk Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">model.pth</td>
      <td><span class="status-badge clean">✓ Clean</span></td>
      <td class="risk-score">12%</td>
    </tr>
  </tbody>
</table>
```

**Features**:
- Subtle header background
- Row hover effects
- Proper spacing and borders
- Responsive scrolling on mobile

---

#### `.status-badge`
Status indicators with color variants.

```html
<span class="status-badge clean">✓ Clean</span>
<span class="status-badge danger">⚠ Threat</span>
```

**Variants**:
- `.clean`: Green background, green text
- `.danger`: Red background, red text
- `.warning`: Orange background, orange text (in globals.css)

**Color Mapping**:
- Clean: 0% - 30% risk
- Warning: 30% - 70% risk  
- Danger: 70%+ risk

---

### Typography

#### `.stat-label`
Small uppercase label text.

```html
<p class="stat-label">Models Scanned</p>
```

**Styling**: 0.75rem, uppercase, secondary color, letter spacing

---

#### `.stat-value`
Large highlighted value with gradient.

```html
<p class="stat-value">24</p>
```

**Styling**: 1.8rem, bold, cyan-magenta gradient text

---

#### `.stat-trend`
Trend indicator text below value.

```html
<p class="stat-trend">+3 this week</p>
```

**Styling**: 0.7rem, muted color, small font

---

#### `.upload-title` & `.upload-subtitle`
Text within upload area.

```html
<p class="upload-title">Click to upload model</p>
<p class="upload-subtitle">Supports .pt, .pth, .onnx</p>
```

---

### Utility Classes

#### `.view-container`
Main content container for full-width views.

```html
<div class="view-container">
  <!-- View content -->
</div>
```

---

#### `.form-group`
Wrapper for form field and label.

```html
<div class="form-group">
  <label>Field Label</label>
  <input class="form-input" />
</div>
```

**Spacing**: 1.5rem margin-bottom

---

#### `.info-card` & `.info-list`
Documentation and feature listing.

```html
<div class="info-card">
  <h3>About This Scan</h3>
  <p>Description text...</p>
  <ul class="info-list">
    <li>Feature 1</li>
    <li>Feature 2</li>
  </ul>
</div>
```

**Features**:
- Proper heading size and spacing
- Reduced text size for descriptions
- Bullet points with cyan indicators

---

## Animation Classes

### `.spin`
Rotation animation for loading spinners.

```html
<Loader2 class="spin" size={20} />
```

**Duration**: 1s, continuous rotation

---

### `.pulsate`
Pulsing glow effect.

```html
<div class="pulsate" />
```

**Duration**: 2s infinite, expands and fades

---

## Responsive Design

### Breakpoints

```css
/* Desktop (1200px+) */
@media (min-width: 1200px) {
  .scan-layout { grid-template-columns: 1fr 350px; }
  .stats-grid { grid-template-columns: repeat(4, 1fr); }
}

/* Tablet (768px - 1200px) */
@media (max-width: 1200px) {
  .scan-layout { grid-template-columns: 1fr; }
}

@media (max-width: 1024px) {
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Mobile (< 768px) */
@media (max-width: 768px) {
  .form-row { grid-template-columns: 1fr; }
  .stats-grid { grid-template-columns: 1fr; }
  .scan-info { position: static; }
}
```

---

## Best Practices

### ✅ Do's
- Use semantic class names for new components
- Group related components with consistent naming
- Update CSS variables for global color changes
- Use responsive grid layouts
- Add focus states for accessibility
- Test across breakpoints

### ❌ Don'ts
- Add inline styles (use CSS classes)
- Hardcode colors (use CSS variables)
- Skip focus states
- Ignore responsive design
- Create single-use classes

---

## Migration Guide

### Old Classes → New Classes

| Old | New | Use Case |
|-----|-----|----------|
| `.input-field` | `.form-input` | Text inputs |
| `.input-field` | `.form-select` | Dropdown selects |
| `.glass-hover` (for upload) | `.upload-trigger` | File upload areas |
| Generic `.card` | `.stat-card` | Dashboard statistics |
| None | `.stats-grid` | Stat card container |
| None | `.scan-layout` | Form + sidebar |

---

## Future Enhancements

Potential additions to the design system:

- [ ] Modal/dialog components
- [ ] Toast notification styles
- [ ] Breadcrumb navigation
- [ ] Pagination components
- [ ] Badge variants
- [ ] Loading skeleton styles
- [ ] Chart color schemes
- [ ] Dark/light mode toggle

---

**Last Updated**: April 5, 2026
**Version**: 1.0 (Production)
**Compatibility**: Modern browsers with CSS Grid, Flexbox, Backdrop Filter support
