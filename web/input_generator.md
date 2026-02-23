# ASCEC Input File Generator

A web-based tool for generating ASCEC v04 input files with an integrated molecular structure converter and 3D visualization.

## Overview

The ASCEC Input File Generator is a standalone HTML application that runs entirely in the browser. It provides:

- **Form-based input** for all ASCEC simulation parameters
- **Molecular structure conversion** from SMILES/IUPAC names to XYZ coordinates
- **3D molecular visualization** using 3Dmol.js
- **Simulation box preview** with dummy atoms at vertices
- **Protocol builder** for multi-stage workflows

## Access

### Online (GitHub Pages)
```
https://manuel2gl.github.io/qft-ascec-similarity/
```

### Local Server
```bash
python serve_web.py

python ascec-v04.py input [port]
```
This starts a local HTTP server (default port 8080) and opens the generator in your browser.

### Standalone
Open `web/ascec_input_generator.html` directly in any modern browser.

---

## Molecular Structure Converter

### PubChem Integration Protocol

The converter uses the **PubChem PUG REST API** to fetch 3D molecular coordinates. It employs a multi-strategy approach to maximize success rate across PubChem's 116+ million compounds.

#### Search Strategies (in order)

1. **SMILES with 3D conformer**
   ```
   https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{input}/SDF?record_type=3d
   ```
   Best for standard SMILES notation (CCO, c1ccccc1, etc.)

2. **Name with 3D conformer**
   ```
   https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input}/SDF?record_type=3d
   ```
   Best for IUPAC names and common names (ethanol, benzene, etc.)

3. **Synonym search → CID → 3D structure**
   ```
   Step 1: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input}/cids/JSON
   Step 2: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d
   ```
   Handles brand names, trade names, and alternative nomenclature (Taxol, Aspirin, Tylenol, etc.)

4. **Name with 2D structure (fallback)**
   ```
   https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input}/SDF
   ```
   Returns 2D coordinates when 3D not available - may need geometry optimization

5. **SMILES with 2D structure (last resort)**
   ```
   https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{input}/SDF
   ```
   Final fallback for edge cases

#### Supported Input Types

| Input Type | Example | Notes |
|------------|---------|-------|
| SMILES | `CCO`, `c1ccccc1` | Standard SMILES notation |
| IUPAC Name | `ethanol`, `benzene` | Systematic names |
| Common Name | `water`, `aspirin` | Everyday names |
| Brand Name | `Taxol`, `Tylenol` | Trade/commercial names |
| CAS Number | Via name search | May work for some compounds |

#### 3D Coordinate Generation

PubChem's 3D coordinates are computed using **OEChem** (OpenEye Scientific Software):
- Conformer generation using OMEGA
- Energy minimization
- Updated periodically (current: PubChem release 2025.09.15)

#### Automatic Centering

All molecules are automatically centered at the origin (0, 0, 0) after retrieval:

```javascript
function centerMolecule(atoms, coords) {
    // Calculate centroid
    let sumX = 0, sumY = 0, sumZ = 0;
    for (const [x, y, z] of coords) {
        sumX += x; sumY += y; sumZ += z;
    }
    const n = coords.length;
    const centroid = [sumX/n, sumY/n, sumZ/n];
    
    // Translate to origin
    return coords.map(([x, y, z]) => [
        x - centroid[0],
        y - centroid[1],
        z - centroid[2]
    ]);
}
```

This ensures molecules are properly positioned for ASCEC's random placement algorithm.

---

## 3D Visualization

### Molecule Preview (3Dmol.js)

When coordinates are generated, a 3D preview appears showing:
- Ball-and-stick representation
- Jmol color scheme (C=gray, O=red, N=blue, H=white, etc.)
- Interactive rotation/zoom
- Spin animation on mouse hover

### Simulation Box Preview

A separate viewer shows the complete simulation setup:
- **Blue edges**: Box boundaries (cube defined by simulation length parameter)
- **Magenta spheres**: Dummy atoms (X) at all 8 box vertices
- **Molecules**: All added molecules centered at origin, overlapping

This mimics ASCEC's `resultbox` output format for visualization.

---

## Input File Format

The generator creates ASCEC v04 `.in` files with this structure:

```
# Line 1: Simulation name
simulation_name
# Line 2: Simulation mode (1=annealing, 2=random configs)
1
# Line 3: Number of configurations (only for mode 2)
10
# Line 4: Simulation cube length (Å)
5.0
# Line 5: Quenching route (1=linear, 2=geometric)
2
# Line 6: Linear params (Ti, ΔT, steps) or geometric (Ti, factor%, steps)
600.0  5.0  100
# Line 7: Max MC cycles per temperature
3000
# Line 8: MC floor value
50
# Line 9: Max displacement, max rotation (Å, radians)
1.0  1.0
# Line 10: Conformational sampling %, max dihedral rotation (°)
30  60
# Line 11: Parallel cores (0 = auto, capped at 12)
0
# Line 12: QM program (1=Gaussian, 2=ORCA, 3=Other)
2
# Line 13: Program alias, Hamiltonian, basis set, charge, multiplicity
orca  HF-3c  _  0  1
*
# Molecule definitions
3  water
O    0.000000    0.000000    0.117300
H    0.756950    0.000000   -0.469200
H   -0.756950    0.000000   -0.469200
*
# Optional protocol block
.in
r3 --box10
calc --critical=0 --retry=3 ../input.inp ../launcher.sh
similarity --th=2 --cores=8
opt --skipped=0 --retry=3 ../input.inp ../launcher.sh
```

---

## Protocol Builder

The protocol builder allows defining multi-stage workflows:

| Stage Type | Description | Common Flags |
|------------|-------------|--------------|
| `rN` | Run N replica annealings | `--box10` |
| `calc` | Single-point calculations | `--critical=0 --retry=3` |
| `similarity` | Cluster similar structures | `--th=2 --cores=8` |
| `opt` | Geometry optimization | `--skipped=0 --retry=3` |

### Example Protocol
```
.in
r3 --box10
calc --critical=0 --retry=3 ../input.inp ../launcher.sh
similarity --th=2 --cores=8
opt --skipped=0 --retry=3 ../input.inp ../launcher.sh
```

---

## Browser Compatibility

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ⚠️ Internet Explorer (not supported)

Requires:
- JavaScript enabled
- WebGL support (for 3D visualization)
- Internet connection (for PubChem API)

---

## Troubleshooting

### "Could not find compound"
- Check spelling of the molecule name
- Try the SMILES notation instead
- Try the IUPAC systematic name
- Search on [PubChem](https://pubchem.ncbi.nlm.nih.gov/) to verify the compound exists

### "Could not parse 3D structure"
- The compound exists but lacks 3D coordinates
- Use Manual Input tab with coordinates from another source (Avogadro, GaussView)

### 3D viewer not showing
- Check WebGL is enabled in your browser
- Try refreshing the page
- Check browser console for errors

### Box viewer shows only vertices
- Add molecules first using either input method
- Click "Refresh" button to update the view

---

## Dependencies

- **3Dmol.js** (v2.4.2) - WebGL molecular viewer
  - CDN: `https://3dmol.org/build/3Dmol-min.js`
  - License: BSD-3-Clause
  
- **PubChem PUG REST API** - Molecular structure database
  - Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
  - No API key required
  - Rate limits: 5 requests/second (generous for interactive use)

---

## Credits

- **ASCEC v04** - Universidad de Antioquia, Química Física Teórica
- **PubChem** - National Center for Biotechnology Information (NCBI)
- **3Dmol.js** - University of Pittsburgh
- **OEChem** - OpenEye Scientific Software (PubChem's 3D coordinate generator)
