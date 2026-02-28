"""
Fabry-Pérot cavity transverse mode visualizer (TEM₀₀ Gaussian beam).

Usage
-----
Set R1, R2, L, and wavelength in the __main__ block at the bottom, then run.
The figure is saved as cavity_mode.png in the working directory.

Sign convention
---------------
R1, R2 > 0  →  concave mirrors facing the cavity interior (normal case)
R1 = R2 → ∞  →  planar (flat) cavity — not a confined Gaussian mode
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── ABCD cavity mode solver ───────────────────────────────────────────────────

def compute_cavity_mode(R1: float, R2: float, L: float,
                        wavelength: float = 1064e-9) -> dict:
    """
    Compute the TEM₀₀ Gaussian mode of a Fabry-Pérot resonator
    using the ABCD (ray-transfer matrix) formalism.

    Parameters
    ----------
    R1, R2    : Radii of curvature [m].  Positive = concave toward cavity.
    L         : Mirror separation [m].
    wavelength: Free-space wavelength [m].

    Returns
    -------
    dict with keys:
      z, w_z          – position array and beam radius along z
      w0, z_waist, zR – waist size, position from M1, Rayleigh range
      w_M1, w_M2      – beam radius at each mirror
      g1, g2          – stability parameters
      L, R1, R2, wavelength
    """
    # Round-trip ABCD matrix (unfolded picture, reference plane just after M1)
    # Sequence of operations: propagate L → reflect M2 → propagate L → reflect M1
    def prop(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    def mirror(R):          # mirror ≡ thin lens, focal length f = R/2
        return np.array([[1.0, 0.0], [-2.0 / R, 1.0]])

    M = mirror(R1) @ prop(L) @ mirror(R2) @ prop(L)
    A, C, D = M[0, 0], M[1, 0], M[1, 1]

    g1, g2 = 1.0 - L / R1, 1.0 - L / R2

    # Stability: eigenvalues of M must lie on unit circle, i.e. |(A+D)/2| ≤ 1
    if abs((A + D) / 2.0) > 1.0 + 1e-9:
        raise ValueError(
            f"Cavity is UNSTABLE  (g₁={g1:.4f}, g₂={g2:.4f}, "
            f"g₁·g₂={g1*g2:.4f}).  Stability requires 0 ≤ g₁·g₂ ≤ 1."
        )
    if abs(C) < 1e-14:
        raise ValueError(
            "Near-planar cavity (C ≈ 0): the Gaussian mode is not confined. "
            "Use finite mirror curvatures."
        )

    # Self-consistency: C·q² + (D−A)·q − B = 0
    # Physical root has Im(q) > 0  (positive Rayleigh range)
    disc = max(0.0, 4.0 - (A + D) ** 2)
    q_M1 = complex((A - D) / (2 * C),
                   np.sqrt(disc) / (2 * abs(C)))
    if q_M1.imag < 0:
        q_M1 = q_M1.conjugate()

    # Beam profile inside the cavity:  q(z) = q_M1 + z  (free-space propagation)
    z = np.linspace(0.0, L, 2000)
    inv_q_z = 1.0 / (q_M1 + z)
    w_z = np.sqrt(-wavelength / (np.pi * inv_q_z.imag))

    idx_w = np.argmin(w_z)
    return dict(
        z=z, w_z=w_z,
        w0=w_z[idx_w], z_waist=z[idx_w],
        zR=np.pi * w_z[idx_w] ** 2 / wavelength,
        w_M1=w_z[0], w_M2=w_z[-1],
        g1=g1, g2=g2,
        L=L, R1=R1, R2=R2, wavelength=wavelength,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cavity(R1: float, R2: float, L: float,
                wavelength: float = 1064e-9) -> dict:
    """Compute and visualise the Fabry-Pérot cavity TEM₀₀ mode."""
    p = compute_cavity_mode(R1, R2, L, wavelength)
    z, w = p['z'], p['w_z']
    w0, z_w, zR = p['w0'], p['z_waist'], p['zR']
    g1, g2 = p['g1'], p['g2']

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    mm = 1e3        # m → mm

    # ── Panel 1: side-view beam envelope (full top row) ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.fill_between(z * mm,  w * mm, -w * mm,
                     alpha=0.25, color='steelblue', label='Beam envelope')
    ax1.plot(z * mm,  w * mm, color='steelblue', lw=2)
    ax1.plot(z * mm, -w * mm, color='steelblue', lw=2)

    # Waist
    ax1.axvline(z_w * mm, color='crimson', ls='--', lw=1.5,
                label=f'Beam waist  (z = {z_w*mm:.1f} mm)')
    ax1.annotate(
        f' w₀ = {w0*1e6:.1f} µm',
        xy=(z_w * mm, w0 * mm),
        xytext=(z_w * mm + L * 0.10 * mm, w0 * mm + w.max() * 0.20 * mm),
        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5),
        fontsize=10, color='crimson',
    )

    # Rayleigh range bracket
    for sign in (-1, 1):
        ax1.axvline((z_w + sign * zR) * mm, color='goldenrod',
                    ls=':', lw=1.2, alpha=0.8)
    mid_y = w0 * mm * np.sqrt(2) * 0.55
    ax1.annotate('', xy=((z_w + zR) * mm, mid_y),
                 xytext=((z_w - zR) * mm, mid_y),
                 arrowprops=dict(arrowstyle='<->', color='goldenrod', lw=1.5))
    ax1.text(z_w * mm, mid_y + w.max() * 0.02 * mm,
             f' 2zᴿ = {2*zR*mm:.1f} mm', fontsize=8, color='goldenrod', va='bottom')

    # Mirror blocks
    mh = w.max() * 1.75 * mm
    ax1.fill_betweenx([-mh, mh], -L * 0.030 * mm, 0,
                      color='slategray', alpha=0.85)
    ax1.fill_betweenx([-mh, mh], L * mm, L * 1.030 * mm,
                      color='slategray', alpha=0.85)
    ax1.text(-L * 0.015 * mm, mh * 0.55, f'M₁\nR₁={R1*mm:.0f} mm',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax1.text(L * 1.015 * mm, mh * 0.55, f'M₂\nR₂={R2*mm:.0f} mm',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax1.set_title(
        f'Fabry-Pérot Resonator — TEM₀₀ Gaussian Mode\n'
        f'R₁ = {R1*mm:.0f} mm,  R₂ = {R2*mm:.0f} mm,  L = {L*mm:.0f} mm,  '
        f'λ = {wavelength*1e9:.0f} nm  │  '
        f'g₁ = {g1:.3f},  g₂ = {g2:.3f},  g₁g₂ = {g1*g2:.4f}',
        fontsize=11,
    )
    ax1.set_xlabel('Position along optical axis  z  (mm)', fontsize=11)
    ax1.set_ylabel('Transverse beam radius  w(z)  (mm)', fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(-L * 0.06 * mm, L * 1.08 * mm)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: 2D intensity map I(r, z) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    r_max = w.max() * 2.5
    r_ax = np.linspace(-r_max, r_max, 400)
    Z2D, R2D = np.meshgrid(z, r_ax)
    W2D = np.interp(Z2D.ravel(), z, w).reshape(Z2D.shape)
    I2D = (w0 / W2D) ** 2 * np.exp(-2 * R2D ** 2 / W2D ** 2)

    im = ax2.pcolormesh(Z2D * mm, R2D * 1e6, I2D,
                        cmap='inferno', shading='auto', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax2, label='Normalised intensity')
    ax2.axvline(z_w * mm, color='white', ls='--', lw=1, alpha=0.6,
                label=f'Waist z={z_w*mm:.1f} mm')
    ax2.set_xlabel('z  (mm)', fontsize=10)
    ax2.set_ylabel('r  (µm)', fontsize=10)
    ax2.set_title('2D Intensity Map  I(r, z)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')

    # ── Panel 3: transverse intensity profiles ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    def plot_profile(ax, w_loc, color, label):
        r = np.linspace(-4 * w_loc, 4 * w_loc, 600)
        I = np.exp(-2 * r ** 2 / w_loc ** 2)
        ax.plot(r * 1e6, I, color=color, lw=2, label=label)
        ax.fill_between(r * 1e6, I, alpha=0.10, color=color)
        for sign in (-1, 1):
            ax.axvline(sign * w_loc * 1e6, color=color, ls='--', lw=1, alpha=0.6)

    plot_profile(ax3, w0,         'steelblue',
                 f'Waist  w₀ = {w0*1e6:.1f} µm  (z = {z_w*mm:.1f} mm)')
    plot_profile(ax3, p['w_M1'], 'seagreen',
                 f'At M₁  w = {p["w_M1"]*1e6:.1f} µm')
    plot_profile(ax3, p['w_M2'], 'darkorange',
                 f'At M₂  w = {p["w_M2"]*1e6:.1f} µm')

    ax3.axhline(np.exp(-2), color='gray', ls=':', lw=1.2,
                label=f'1/e² ≈ {np.exp(-2):.3f}')
    ax3.set_xlabel('Radial position  r  (µm)', fontsize=10)
    ax3.set_ylabel('Normalised intensity', fontsize=10)
    ax3.set_title('Transverse Intensity Profiles', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.10)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: stability diagram ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])

    g_lin = np.linspace(-1.6, 1.6, 600)
    G1, G2 = np.meshgrid(g_lin, g_lin)
    prod = G1 * G2

    ax4.contourf(G1, G2, np.clip(prod, -0.05, 1.05),
                 levels=[-0.04, 0, 1, 1.04],
                 colors=['#ffcccc', '#c8eac8', '#ffcccc'], alpha=0.60)
    ax4.contour(G1, G2, prod, levels=[0, 1],
                colors=['darkgreen'], linewidths=2)

    specials = {
        'Planar\n(1,1)':          (1,  1),
        'Concentric\n(−1,−1)':   (-1, -1),
        'Confocal\n(0,0)':        (0,  0),
        'Hemi-sph.\n(0,1)':       (0,  1),
        'Hemi-sph.\n(1,0)':       (1,  0),
    }
    for name, (gx, gy) in specials.items():
        ax4.plot(gx, gy, 'k^', ms=6, zorder=4)
        ax4.annotate(name, (gx, gy), xytext=(4, 4),
                     textcoords='offset points', fontsize=6.5)

    ax4.plot(g1, g2, 'r*', ms=15, zorder=5,
             label=f'This cavity\n({g1:.3f}, {g2:.3f})')
    ax4.axhline(0, color='k', lw=0.5)
    ax4.axvline(0, color='k', lw=0.5)
    ax4.set_xlabel('g₁ = 1 − L/R₁', fontsize=10)
    ax4.set_ylabel('g₂ = 1 − L/R₂', fontsize=10)
    ax4.set_title('Cavity Stability Diagram', fontsize=10)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.set_xlim(-1.6, 1.6)
    ax4.set_ylim(-1.6, 1.6)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, 0.5, 'STABLE', transform=ax4.transAxes,
             ha='center', va='center', fontsize=10, color='darkgreen',
             alpha=0.5, fontweight='bold')

    plt.suptitle('Fabry-Pérot Optical Resonator — Gaussian Mode Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = 'cavity_mode.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {out}')
    plt.show()

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print("  FABRY-PÉROT CAVITY — TEM₀₀ MODE PARAMETERS")
    print(f"{'═'*52}")
    print(f"  Cavity length  L       {L*mm:>10.2f} mm")
    print(f"  ROC mirror 1   R1      {R1*mm:>10.2f} mm")
    print(f"  ROC mirror 2   R2      {R2*mm:>10.2f} mm")
    print(f"  Wavelength     λ       {wavelength*1e9:>10.1f} nm")
    print(f"  {'─'*46}")
    print(f"  g₁ = 1 − L/R₁         {g1:>10.4f}")
    print(f"  g₂ = 1 − L/R₂         {g2:>10.4f}")
    print(f"  Stability  g₁·g₂       {g1*g2:>10.4f}   (need 0 ≤ · ≤ 1)")
    print(f"  {'─'*46}")
    print(f"  Beam waist  w₀         {w0*1e6:>10.2f} µm")
    print(f"  Waist pos. from M1     {z_w*mm:>10.2f} mm")
    print(f"  Rayleigh range  zᴿ     {zR*mm:>10.2f} mm")
    print(f"  Beam radius at M1      {p['w_M1']*1e6:>10.2f} µm")
    print(f"  Beam radius at M2      {p['w_M2']*1e6:>10.2f} µm")
    print(f"{'═'*52}")

    return p


# ── Configuration — edit these values ────────────────────────────────────────
if __name__ == '__main__':
    R1         = 0.500     # mirror 1 radius of curvature  [m]
    R2         = 0.500     # mirror 2 radius of curvature  [m]
    L          = 0.400     # cavity length (mirror separation)  [m]
    wavelength = 1064e-9   # wavelength  [m]  (1064 nm = Nd:YAG)

    plot_cavity(R1, R2, L, wavelength)
