// OverlayGL.h
// Small helpers for the viewer overlays (OFX overlay interacts): legacy OpenGL drawing in
// canonical coordinates, with a dark shadow under every line so that the overlay stays
// readable on any image. Sizes given in screen pixels are converted with the pixel scale.
#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "ofxsInteract.h"

#include <cmath>

namespace ovl {

struct Colour { float r, g, b; };

// Default overlay colour (host suggestion when available), and the hover / drag highlight.
inline Colour baseColour(const OFX::Interact& i)
{
    OfxRGBColourD c;
    if (i.getSuggestedColour(c)) return { (float)c.r, (float)c.g, (float)c.b };
    return { 0.8f, 0.8f, 0.8f };
}
inline Colour highlight() { return { 1.f, 0.85f, 0.25f }; }

// Shadow offset of one screen pixel, down-right.
struct Pen {
    double sx, sy;                                    // canonical units per screen pixel
    explicit Pen(const OfxPointD& pixelScale) : sx(pixelScale.x), sy(pixelScale.y) {}
    double hitRadius() const { return 8.0; }          // screen pixels
    // distance between two canonical points, in screen pixels
    double screenDist(double ax, double ay, double bx, double by) const
    {
        const double dx = (ax - bx) / sx, dy = (ay - by) / sy;
        return std::sqrt(dx * dx + dy * dy);
    }
};

// Calls draw(colour, dx, dy) twice: shadow, then the colour.
template <class F>
void withShadow(const Pen& pen, const Colour& c, F&& draw)
{
    glColor3f(0.f, 0.f, 0.f);
    draw(pen.sx, -pen.sy);
    glColor3f(c.r, c.g, c.b);
    draw(0.0, 0.0);
}

struct GLState {
    GLState()
    {
        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    ~GLState() { glPopAttrib(); }
};

// Circle (radii rx, ry in canonical units), optionally dashed.
inline void circle(const Pen& pen, const Colour& c, double cx, double cy, double rx, double ry, float width = 1.5f,
                   bool dashed = false)
{
    if (rx <= 0.0 || ry <= 0.0) return;
    const double rScreen = std::fmax(rx / pen.sx, ry / pen.sy);
    const int n = (int)std::fmin(512.0, std::fmax(24.0, rScreen * 0.75));
    glLineWidth(width);
    withShadow(pen, c, [&](double dx, double dy) {
        glBegin(dashed ? GL_LINES : GL_LINE_LOOP);
        const int m = dashed ? (n / 2) * 2 : n;
        for (int k = 0; k < m; ++k) {
            const double a = 2.0 * 3.14159265358979323846 * k / m;
            glVertex2d(cx + rx * std::cos(a) + dx, cy + ry * std::sin(a) + dy);
        }
        glEnd();
    });
}

inline void line(const Pen& pen, const Colour& c, double ax, double ay, double bx, double by, float width = 1.5f)
{
    glLineWidth(width);
    withShadow(pen, c, [&](double dx, double dy) {
        glBegin(GL_LINES);
        glVertex2d(ax + dx, ay + dy);
        glVertex2d(bx + dx, by + dy);
        glEnd();
    });
}

// Arrow from a to b, head size in screen pixels.
inline void arrow(const Pen& pen, const Colour& c, double ax, double ay, double bx, double by, float width = 2.f)
{
    line(pen, c, ax, ay, bx, by, width);
    const double vx = (bx - ax) / pen.sx, vy = (by - ay) / pen.sy;   // screen space
    const double len = std::sqrt(vx * vx + vy * vy);
    if (len < 1.0) return;
    const double ux = vx / len, uy = vy / len, h = std::fmin(14.0, len * 0.4);
    for (int s = -1; s <= 1; s += 2) {
        const double cs = std::cos(0.45), sn = std::sin(0.45) * s;
        const double hx = -(ux * cs - uy * sn) * h, hy = -(ux * sn + uy * cs) * h;
        line(pen, c, bx, by, bx + hx * pen.sx, by + hy * pen.sy, width);
    }
}

// Square handle of `size` screen pixels.
inline void handle(const Pen& pen, const Colour& c, double x, double y, double size = 7.0)
{
    glPointSize((float)size);
    withShadow(pen, c, [&](double dx, double dy) {
        glBegin(GL_POINTS);
        glVertex2d(x + dx, y + dy);
        glEnd();
    });
}

// Cross (centre marker) of `size` screen pixels.
inline void cross(const Pen& pen, const Colour& c, double x, double y, double size = 8.0)
{
    line(pen, c, x - size * pen.sx, y, x + size * pen.sx, y);
    line(pen, c, x, y - size * pen.sy, x, y + size * pen.sy);
}

inline void rect(const Pen& pen, const Colour& c, double x1, double y1, double x2, double y2, float width = 1.5f)
{
    glLineWidth(width);
    withShadow(pen, c, [&](double dx, double dy) {
        glBegin(GL_LINE_LOOP);
        glVertex2d(x1 + dx, y1 + dy);
        glVertex2d(x2 + dx, y1 + dy);
        glVertex2d(x2 + dx, y2 + dy);
        glVertex2d(x1 + dx, y2 + dy);
        glEnd();
    });
}

} // namespace ovl
