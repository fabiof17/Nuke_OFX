// ColourSpaceView.h
// MasterKeyer_OFX viewer overlay: a 3D view of the colour space of the 3D matte, drawn in a
// corner of the viewer (screen space). Shows the axes (luma x Luma Weight vertical, Cb and Cr
// horizontal), the neutral (grey) axis, the Primary and Secondary samples, the tolerance volume
// and the softness envelope (deformed by Gain and Shadows / Midtones / Highlights).
// Drag in the panel to rotate, Shift + drag to zoom, double-click to reset the view; drag the
// handle of its inner corner to resize it. Colours darker than black (luma < 0) do not exist:
// the volumes are cut at the black level, shown as a floor.
#pragma once

#include "OverlayGL.h"
#include "MasterKeyerCommon.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace csv {

struct V3 { double x, y, z; };
inline V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline V3 operator*(V3 a, double k) { return { a.x * k, a.y * k, a.z * k }; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline double len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a) { const double l = len(a); return l > 1e-12 ? a * (1.0 / l) : V3{ 0, 1, 0 }; }

// World position of a colour: x = Cb, y = luma x weight (up), z = Cr.
inline V3 world(const float* rgb, float lw)
{
    float v[3];
    mk::to3D(rgb, lw, v);
    return { v[1], v[0], v[2] };
}

// Window (viewport) <-> canonical conversions, captured at each draw.
struct Frame {
    double mv[16] = { 0 }, pj[16] = { 0 };
    int vp[4] = { 0, 0, 1, 1 };
    bool valid = false;
    void capture()
    {
        glGetDoublev(GL_MODELVIEW_MATRIX, mv);
        glGetDoublev(GL_PROJECTION_MATRIX, pj);
        glGetIntegerv(GL_VIEWPORT, vp);
        valid = vp[2] > 0 && vp[3] > 0;
    }
    // canonical (x, y) -> window pixels
    bool toWindow(double x, double y, double& wx, double& wy) const
    {
        if (!valid) return false;
        double e[4], c[4];
        for (int i = 0; i < 4; ++i) e[i] = mv[i] * x + mv[4 + i] * y + mv[12 + i];
        for (int i = 0; i < 4; ++i) c[i] = pj[i] * e[0] + pj[4 + i] * e[1] + pj[8 + i] * e[2] + pj[12 + i] * e[3];
        if (std::fabs(c[3]) < 1e-12) return false;
        wx = vp[0] + (c[0] / c[3] + 1.0) * 0.5 * vp[2];
        wy = vp[1] + (c[1] / c[3] + 1.0) * 0.5 * vp[3];
        return true;
    }
};

// Settings the view needs, read from the parameters.
struct Settings {
    mk::Params p;          // primary, secondary, mix, tolerance, softness3D, lumaWeight, gain, tones
    int corner = 3;        // 0 bottom left, 1 bottom right, 2 top left, 3 top right
    double size = 260.0;   // panel size, screen pixels
    bool tolShaded = false; // tolerance volume: wireframe or Gouraud-shaded
    bool envShaded = false; // softness envelope: wireframe or Gouraud-shaded
    bool rgbView = false;   // display space: Y Cb Cr (false) or the RGB cube
    double yaw = 0.6, pitch = 0.35, zoom = 1.0;
};

class View {
public:
    // Panel rectangle in window pixels.
    void layout(const Frame& f, int corner, double wanted)
    {
        const double vw = f.vp[2], vh = f.vp[3], m = 12.0;
        size = std::max(120.0, std::min(wanted, std::min(vw, vh) - 2 * m));
        right = corner == 1 || corner == 3;
        top = corner == 2 || corner == 3;
        x0 = right ? f.vp[0] + vw - m - size : f.vp[0] + m;
        y0 = top ? f.vp[1] + vh - m - size : f.vp[1] + m;
    }
    bool inside(double wx, double wy) const { return wx >= x0 && wx <= x0 + size && wy >= y0 && wy <= y0 + size; }
    // Resize handle: the corner of the panel towards the centre of the viewer.
    void handlePos(double& hx, double& hy) const { hx = right ? x0 : x0 + size; hy = top ? y0 : y0 + size; }
    bool onHandle(double wx, double wy) const
    {
        double hx, hy;
        handlePos(hx, hy);
        return std::fabs(wx - hx) <= 14.0 && std::fabs(wy - hy) <= 14.0;
    }
    // New size when the handle is dragged by (dx, dy) window pixels.
    double resized(double size0, double dx, double dy) const
    {
        const double sx = right ? -dx : dx, sy = top ? -dy : dy;
        return std::max(120.0, std::min(2000.0, size0 + 0.5 * (sx + sy)));
    }

    void draw(const Settings& s, bool highlight, bool handleHot = false)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glOrtho(vp[0], vp[0] + vp[2], vp[1], vp[1] + vp[3], -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // panel
        glColor4f(0.08f, 0.08f, 0.09f, 0.82f);
        glBegin(GL_QUADS);
        glVertex2d(x0, y0); glVertex2d(x0 + size, y0); glVertex2d(x0 + size, y0 + size); glVertex2d(x0, y0 + size);
        glEnd();
        glLineWidth(1.f);
        glColor4f(highlight ? 1.f : 0.45f, highlight ? 0.85f : 0.45f, highlight ? 0.25f : 0.45f, 1.f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(x0, y0); glVertex2d(x0 + size, y0); glVertex2d(x0 + size, y0 + size); glVertex2d(x0, y0 + size);
        glEnd();

        // scissor to the panel
        glPushAttrib(GL_SCISSOR_BIT);
        glEnable(GL_SCISSOR_TEST);
        glScissor((GLint)x0, (GLint)y0, (GLsizei)size, (GLsizei)size);
        drawSpace(s);
        glPopAttrib();

        // resize handle: a small triangle in the inner corner
        double hx, hy;
        handlePos(hx, hy);
        const double dx = right ? 12.0 : -12.0, dy = top ? 12.0 : -12.0;
        glColor4f(handleHot ? 1.f : 0.6f, handleHot ? 0.85f : 0.6f, handleHot ? 0.25f : 0.6f, 0.9f);
        glBegin(GL_TRIANGLES);
        glVertex2d(hx, hy); glVertex2d(hx + dx, hy); glVertex2d(hx, hy + dy);
        glEnd();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }

    double x0 = 0, y0 = 0, size = 0;
    bool right = true, top = true;

private:
    // ---------------------------------------------------------------- geometry
    struct Geo {
        V3 a, b;            // segment of the tolerance volume (world)
        double tol, soft;
        float lw;
        const mk::Params* p;
    };

    // Gain of the envelope at a world point (luma recovered from y / weight).
    static double gainAt(const Geo& g, V3 q, double lumaFallback)
    {
        const double y = g.lw > 1e-6f ? q.y / g.lw : lumaFallback;
        const float c[3] = { (float)y, (float)y, (float)y };
        return mk::matteGain(*g.p, c);
    }

    // Radius of the envelope along direction n from axis point q: the first r where the matte
    // reaches 1, i.e. r = tol + soft x gain(q + n r) (bisection: the gain varies with the luma).
    static double envelope(const Geo& g, V3 q, V3 n, double lumaFallback)
    {
        double hi = g.tol;
        for (double y : { 0.0, 0.35, 0.6, 1.0 }) {
            const float c[3] = { (float)y, (float)y, (float)y };
            hi = std::max(hi, g.tol + g.soft * mk::matteGain(*g.p, c));
        }
        double lo = 0.0;
        for (int it = 0; it < 30; ++it) {
            const double r = 0.5 * (lo + hi);
            if (r - g.tol - g.soft * gainAt(g, q + n * r, lumaFallback) < 0.0) lo = r; else hi = r;
        }
        return hi;
    }

    // Surface point of a capsule, parameters u in [0, 1] along (cap A, cylinder, cap B), phi.
    static V3 surface(const Geo& g, double u, double phi, bool env, double lumaFallback, const V3& ax, const V3& e1,
                      const V3& e2)
    {
        V3 q, n;
        const V3 radial = e1 * std::cos(phi) + e2 * std::sin(phi);
        if (u < 1.0 / 3.0) {                       // cap A: theta 0 .. pi/2
            const double th = u * 3.0 * 1.5707963267948966;
            q = g.a;
            n = ax * (-std::cos(th)) + radial * std::sin(th);
        } else if (u <= 2.0 / 3.0) {               // cylinder
            q = g.a + (g.b - g.a) * ((u - 1.0 / 3.0) * 3.0);
            n = radial;
        } else {                                   // cap B: theta pi/2 .. 0
            const double th = (1.0 - u) * 3.0 * 1.5707963267948966;
            q = g.b;
            n = ax * std::cos(th) + radial * std::sin(th);
        }
        const double r = env ? envelope(g, q, n, lumaFallback) : g.tol;
        return q + n * r;
    }

    // ----------------------------------------------------------------- drawing
    void drawSpace(const Settings& s)
    {
        const mk::Params& p = s.p;
        float bm[3];
        for (int i = 0; i < 3; ++i) bm[i] = p.primary[i] + (p.secondary[i] - p.primary[i]) * mk::diClamp01(p.mix);
        Geo g;
        g.a = world(p.primary, p.lumaWeight);
        g.b = world(bm, p.lumaWeight);
        g.tol = std::max(0.f, p.tolerance);
        g.soft = std::max(0.f, p.softness3D);
        g.lw = p.lumaWeight;
        g.p = &p;
        const double lumaFallback = mk::diLuma(p.primary);

        // axis of the capsule and a basis around it
        V3 ax = g.b - g.a;
        const bool seg = len(ax) > 1e-9;
        ax = seg ? norm(ax) : V3{ 0, 1, 0 };
        V3 e1 = norm(cross(ax, std::fabs(ax.y) < 0.9 ? V3{ 0, 1, 0 } : V3{ 1, 0, 0 }));
        V3 e2 = cross(ax, e1);

        // view: centre on the volume, scale to fit the envelope
        rgb = s.rgbView;
        lwInv = 1.0 / std::max(0.05, (double)p.lumaWeight);
        const V3 centre = (g.a + g.b) * 0.5;
        double extent = 0.5 * len(g.b - g.a) + g.tol;
        for (double y : { 0.0, 0.35, 0.6, 1.0 }) {
            const float c[3] = { (float)y, (float)y, (float)y };
            extent = std::max(extent, 0.5 * len(g.b - g.a) + g.tol + g.soft * mk::matteGain(p, c));
        }
        extent = std::max(extent, 0.05);
        // extent in the display space (RGB: the change of space stretches the volumes)
        // RGB: the whole cube is framed (centre grey 0.5, half-diagonal ~0.87), so that the
        // volumes are seen where they are among all the colours
        const double dispExtent = rgb ? 0.9 : extent;
        scale = size * 0.42 / dispExtent * s.zoom;
        cx = x0 + size * 0.5; cy = y0 + size * 0.5;
        cyaw = std::cos(s.yaw); syaw = std::sin(s.yaw); cpit = std::cos(s.pitch); spit = std::sin(s.pitch);
        origin = rgb ? V3{ 0.5, 0.5, 0.5 } : centre;

        if (rgb) drawCube();
        else     drawYCC(centre, extent);

        // tolerance volume (yellow) inside, then softness envelope (cyan, translucent), cut at black
        clipBlack = true;
        if (s.tolShaded) shaded(g, false, lumaFallback, ax, e1, e2, 1.f, 0.82f, 0.2f, 1.f);
        else             wire(g, false, lumaFallback, ax, e1, e2, 1.f, 0.85f, 0.25f, 0.95f);
        if (s.envShaded) shaded(g, true, lumaFallback, ax, e1, e2, 0.25f, 0.8f, 1.f, 0.38f);
        else             wire(g, true, lumaFallback, ax, e1, e2, 0.25f, 0.85f, 1.f, 0.45f);
        clipBlack = false;

        // samples
        point3(g.a, p.primary, 9.f);
        if (seg) point3(world(p.secondary, p.lumaWeight), p.secondary, 7.f);
    }

    // RGB cube: edges of [0, 1]^3, axes R, G, B from black, neutral diagonal (dashed).
    void drawCube()
    {
        glLineWidth(1.f);
        glColor4f(0.5f, 0.5f, 0.5f, 0.45f);
        glBegin(GL_LINES);
        for (int a = 0; a < 3; ++a)
            for (int k = 0; k < 4; ++k) {
                double u[3] = { 0, 0, 0 }, v[3];
                u[(a + 1) % 3] = k & 1; u[(a + 2) % 3] = (k >> 1) & 1;
                if (u[(a + 1) % 3] == 0 && u[(a + 2) % 3] == 0) continue;   // the axes are drawn below
                for (int i = 0; i < 3; ++i) v[i] = u[i];
                v[a] = 1;
                vertexDisp(V3{ u[0], u[1], u[2] }); vertexDisp(V3{ v[0], v[1], v[2] });
            }
        glEnd();
        const float col[3][3] = { { 1.f, 0.3f, 0.3f }, { 0.3f, 1.f, 0.35f }, { 0.35f, 0.5f, 1.f } };
        for (int a = 0; a < 3; ++a) {
            V3 e{ 0, 0, 0 };
            (a == 0 ? e.x : a == 1 ? e.y : e.z) = 1.0;
            glLineWidth(1.5f);
            glColor4f(col[a][0], col[a][1], col[a][2], 1.f);
            glBegin(GL_LINES);
            vertexDisp(V3{ 0, 0, 0 }); vertexDisp(e);
            glEnd();
        }
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0x3333);
        glLineWidth(1.f);
        glColor4f(0.6f, 0.6f, 0.6f, 0.7f);
        glBegin(GL_LINES);
        vertexDisp(V3{ 0, 0, 0 }); vertexDisp(V3{ 1, 1, 1 });
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    // Y Cb Cr: axes from the centre of the volume, neutral axis, floor at black.
    void drawYCC(const V3& centre, double extent)
    {
        // axes from the centre: luma (white, up), Cb (blue), Cr (red)
        const double L = extent * 1.1;
        line3(centre, centre + V3{ 0, L, 0 }, 0.85f, 0.85f, 0.85f, 1.f, 1.5f);
        line3(centre, centre + V3{ L, 0, 0 }, 0.35f, 0.5f, 1.f, 1.f, 1.5f);
        line3(centre, centre + V3{ 0, 0, L }, 1.f, 0.35f, 0.3f, 1.f, 1.5f);
        // neutral (grey) axis: Cb = Cr = 0
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0x3333);
        line3(V3{ 0, centre.y - L, 0 }, V3{ 0, centre.y + L, 0 }, 0.6f, 0.6f, 0.6f, 0.7f, 1.f);
        glDisable(GL_LINE_STIPPLE);

        // black level (luma 0): a floor grid, drawn when the envelope reaches it
        if (centre.y - extent < 0.0) {
            const double h = L;
            glColor4f(0.5f, 0.5f, 0.5f, 0.35f);
            glLineWidth(1.f);
            glBegin(GL_LINES);
            for (int k = -4; k <= 4; ++k) {
                const double o = h * k / 4.0;
                vertex(V3{ centre.x + o, 0, centre.z - h }); vertex(V3{ centre.x + o, 0, centre.z + h });
                vertex(V3{ centre.x - h, 0, centre.z + o }); vertex(V3{ centre.x + h, 0, centre.z + o });
            }
            glEnd();
        }
    }

    void wire(const Geo& g, bool env, double lumaFallback, const V3& ax, const V3& e1, const V3& e2, float r, float gg,
              float b, float a)
    {
        const int meridians = 16, steps = 36, rings = 7;
        glColor4f(r, gg, b, a);
        glLineWidth(env ? 1.f : 1.5f);
        for (int m = 0; m < meridians; ++m) {
            const double phi = 6.283185307179586 * m / meridians;
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= steps; ++i) vertex(surface(g, (double)i / steps, phi, env, lumaFallback, ax, e1, e2));
            glEnd();
        }
        for (int k = 1; k < rings; ++k) {
            const double u = (double)k / rings;
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < 48; ++i)
                vertex(surface(g, u, 6.283185307179586 * i / 48, env, lumaFallback, ax, e1, e2));
            glEnd();
        }
    }

    // Gouraud shading: the surface is cut into quads, lit per vertex (Lambert + specular, light
    // from the upper left of the viewer, two-sided) and drawn back to front (painter's algorithm,
    // no depth buffer) with the colours interpolated across each quad.
    void shaded(const Geo& g, bool env, double lumaFallback, const V3& ax, const V3& e1, const V3& e2, float r,
                float gg, float b, float a)
    {
        const int nu = 40, nphi = 36;
        std::vector<V3> P((size_t)(nu + 1) * nphi), Q((size_t)(nu + 1) * nphi);
        for (int i = 0; i <= nu; ++i)
            for (int j = 0; j < nphi; ++j) {
                const double u = (double)i / nu, phi = 6.283185307179586 * j / nphi;
                V3 q = surface(g, u, phi, env, lumaFallback, ax, e1, e2);
                if (q.y < 0.0) q.y = 0.0;                     // cut at black
                P[(size_t)i * nphi + j] = toDisp(q);          // normals, depth and drawing in display space
                // matching point of the axis, to orient the normals outwards
                Q[(size_t)i * nphi + j] = toDisp(u < 1.0 / 3.0 ? g.a : (u > 2.0 / 3.0 ? g.b : g.a + (g.b - g.a) * ((u - 1.0 / 3.0) * 3.0)));
            }
        auto at = [&](int i, int j) -> const V3& { return P[(size_t)i * nphi + ((j % nphi) + nphi) % nphi]; };
        std::vector<V3> N(P.size());
        for (int i = 0; i <= nu; ++i)
            for (int j = 0; j < nphi; ++j) {
                const V3 du = at(std::min(i + 1, nu), j) - at(std::max(i - 1, 0), j);
                const V3 dp = at(i, j + 1) - at(i, j - 1);
                const V3& p = at(i, j);
                const V3 out = p - Q[(size_t)i * nphi + j];
                V3 n = cross(du, dp);
                if (len(n) < 1e-12) n = out;
                if (!rgb && p.y <= 1e-9 && out.y < 0.0) n = V3{ 0, -1, 0 };   // flat floor where cut
                if (dot(n, out) < 0.0) n = n * -1.0;
                N[(size_t)i * nphi + j] = norm(n);
            }
        struct Quad { double depth; int i, j; };
        std::vector<Quad> quads;
        quads.reserve((size_t)nu * nphi);
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < nphi; ++j)
                quads.push_back({ 0.25 * (depth(at(i, j)) + depth(at(i + 1, j)) + depth(at(i + 1, j + 1)) + depth(at(i, j + 1))), i, j });
        std::sort(quads.begin(), quads.end(), [](const Quad& x, const Quad& y) { return x.depth < y.depth; });

        glPushAttrib(GL_LIGHTING_BIT);
        glShadeModel(GL_SMOOTH);
        glBegin(GL_QUADS);
        for (const Quad& qd : quads) {
            const int ii[4] = { qd.i, qd.i + 1, qd.i + 1, qd.i };
            const int jj[4] = { qd.j, qd.j, qd.j + 1, qd.j + 1 };
            for (int k = 0; k < 4; ++k) {
                const int jw = jj[k] % nphi;
                float c[3];
                light(N[(size_t)ii[k] * nphi + jw], r, gg, b, c);
                glColor4f(c[0], c[1], c[2], a);
                vertexDisp(P[(size_t)ii[k] * nphi + jw]);
            }
        }
        glEnd();
        glPopAttrib();
    }

    // Lit colour of a vertex of normal n (world).
    void light(V3 n, float r, float g, float b, float* c) const
    {
        const double nx = n.x * cyaw + n.z * syaw, nz1 = -n.x * syaw + n.z * cyaw;
        double vy = n.y * cpit - nz1 * spit, vz = n.y * spit + nz1 * cpit, vx = nx;
        if (vz < 0.0) { vx = -vx; vy = -vy; vz = -vz; }          // two-sided
        const V3 L = norm(V3{ -0.35, 0.55, 0.76 }), H = norm(L + V3{ 0, 0, 1 });
        const V3 v{ vx, vy, vz };
        const double diff = std::max(0.0, dot(v, L));
        const double spec = 0.35 * std::pow(std::max(0.0, dot(v, H)), 24.0);
        const double k = 0.22 + 0.78 * diff;
        c[0] = (float)std::min(1.0, r * k + spec);
        c[1] = (float)std::min(1.0, g * k + spec);
        c[2] = (float)std::min(1.0, b * k + spec);
    }

    // Display space of a world point (x = Cb, y = luma x weight, z = Cr): itself, or (R, G, B).
    V3 toDisp(V3 w) const
    {
        if (!rgb) return w;
        const double Y = w.y * lwInv, R = w.z + Y, B = w.x + Y;
        return { R, (Y - 0.2126 * R - 0.0722 * B) / 0.7152, B };
    }

    // Distance towards the viewer (larger = nearer), display space.
    double depth(V3 q) const
    {
        const V3 d = q - origin;
        const double z1 = -d.x * syaw + d.z * cyaw;
        return d.y * spit + z1 * cpit;
    }

    void project(V3 q, double& px, double& py) const
    {
        const V3 d = q - origin;
        const double x1 = d.x * cyaw + d.z * syaw, z1 = -d.x * syaw + d.z * cyaw;
        const double y1 = d.y * cpit - z1 * spit;
        px = cx + x1 * scale;
        py = cy + y1 * scale;
    }
    void vertex(V3 q) const
    {
        if (clipBlack && q.y < 0.0) q.y = 0.0;    // no colour is darker than black
        vertexDisp(toDisp(q));
    }
    void vertexDisp(V3 d) const
    {
        double x, y;
        project(d, x, y);
        glVertex2d(x, y);
    }
    void line3(V3 a, V3 b, float r, float g, float bl, float al, float w) const
    {
        glLineWidth(w);
        glColor4f(r, g, bl, al);
        glBegin(GL_LINES);
        vertex(a); vertex(b);
        glEnd();
    }
    // sample point drawn in its own colour (normalised so that it is visible), black outline
    void point3(V3 q, const float* rgb, float sz) const
    {
        const float m = std::max(1e-6f, std::max(rgb[0], std::max(rgb[1], rgb[2])));
        glPointSize(sz + 2.f);
        glColor4f(0.f, 0.f, 0.f, 1.f);
        glBegin(GL_POINTS); vertex(q); glEnd();
        glPointSize(sz);
        glColor4f(rgb[0] / m, rgb[1] / m, rgb[2] / m, 1.f);
        glBegin(GL_POINTS); vertex(q); glEnd();
    }

    double scale = 1, cx = 0, cy = 0, cyaw = 1, syaw = 0, cpit = 1, spit = 0;
    bool clipBlack = false, rgb = false;
    double lwInv = 2.0;
    V3 origin{ 0, 0, 0 };
};

} // namespace csv
