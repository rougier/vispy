#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Quiver-like using only gloo """

import numpy as np
from vispy import app
from vispy import gloo
from vispy.util.transforms import ortho
from vispy.gloo import Program, VertexBuffer


vertex = """
#version 120

// Constants
// ------------------------------------
const float SQRT_2 = 1.4142135623730951;

// Uniform
// ------------------------------------
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_antialias;

// Attributes
// ------------------------------------
attribute float a_size;
attribute float a_orientation;
attribute vec3  a_position;
attribute float a_linewidth;
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;

// Varyings
// ------------------------------------
varying float v_linewidth;
varying float v_size;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
varying vec2  v_rotation;


// Main
// ------------------------------------
void main (void)
{
    v_size = a_size;
    v_linewidth = a_linewidth;
    v_fg_color = a_fg_color;
    v_bg_color = a_bg_color;
    v_rotation = vec2(cos(a_orientation), sin(a_orientation));
    gl_Position = u_projection  * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = SQRT_2 * v_size + 2 * (a_linewidth + 1.5*u_antialias);
}
"""

fragment = """
#version 120

float arrow(vec2 P, float size)
{
    const float SQRT_2 = 1.4142135623730951;
    float x = -P.y;
    float y = P.x;

    float r1 = abs(x) + abs(y) - size/2;
    float r2 = max(abs(x+size/2), abs(y)) - size/2;
    float r3 = max(abs(x-size/6)-size/4, abs(y)- size/4);
    return min(r3,max(.75*r1,r2));
}

float arrow2(vec2 P, float size)
{
    const float SQRT_2 = 1.4142135623730951;
    float x = -1/SQRT_2 * (P.y - P.x);
    float y = -1/SQRT_2 * (P.y + P.x);

    float r1 = max(abs(x),        abs(y))        - size/3;
    float r2 = max(abs(x-size/5), abs(y-size/5)) - size/3;
    float r3 = max(abs(P.y+size/16)- size/3, abs(P.x)- size/10);

    return min(r3,max(r1,-r2));
}


// Constants
// ------------------------------------
const float SQRT_2 = 1.4142135623730951;

// External functions
// ------------------------------------
float marker(vec2 P, float size);

// Uniforms
// ------------------------------------
uniform float u_antialias;

// Varyings
// ------------------------------------
varying vec4  v_fg_color;
varying vec4  v_bg_color;
varying float v_linewidth;
varying float v_size;
varying vec2  v_rotation;

// Main
// ------------------------------------
void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    P = vec2(v_rotation.x*P.x - v_rotation.y*P.y,
             v_rotation.y*P.x + v_rotation.x*P.y);

    float point_size = SQRT_2*v_size  + 2 * (v_linewidth + 1.5*u_antialias);
    float t = v_linewidth/2.0 - u_antialias;

    float signed_distance = arrow2(P*point_size, v_size);

    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/u_antialias;
    alpha = exp(-alpha*alpha);

    // Within linestroke
    if( border_distance < 0 )
        gl_FragColor = v_fg_color;
    else if( signed_distance < 0 )
        // Inside shape
        if( border_distance > (v_linewidth/2.0 + u_antialias) )
            gl_FragColor = v_bg_color;
        else // Line stroke interior border
            gl_FragColor = mix(v_bg_color,v_fg_color,alpha);
    else
        // Outide shape
        if( border_distance > (v_linewidth/2.0 + u_antialias) )
            discard;
        else // Line stroke exterior border
            gl_FragColor = vec4(v_fg_color.rgb, v_fg_color.a * alpha);
}
"""



n = 500
data = np.zeros(n, dtype=[('a_position',    np.float32, 3),
                          ('a_fg_color',    np.float32, 4),
                          ('a_bg_color',    np.float32, 4),
                          ('a_size',        np.float32, 1),
                          ('a_orientation', np.float32, 1),
                          ('a_linewidth',   np.float32, 1)])
data['a_fg_color'] = 0, 0, 0, 1
data['a_bg_color'] = 0, 0, 0, 1
data['a_linewidth'] = 1
u_antialias = 1

radius, theta, dtheta = 255.0, 0.0, 5.5 / 180.0 * np.pi
for i in range(500):
    theta += dtheta
    x = 256 + radius * np.cos(theta)
    y = 256 + radius * np.sin(theta)
    r = 10.1 - i * 0.02
    radius -= 0.45
    data['a_position'][i] = x, y, 0
    data['a_size'][i] = 2*r
    data['a_orientation'][i] = theta


class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')

        # This size is used for comparison with agg (via matplotlib)
        self.size = 512, 512
        self.title = "Markers demo [press space to change marker]"

        self.vbo = VertexBuffer(data)
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = ortho(0, self.size[0], 0, self.size[1], -1, 1)

        program = Program(vertex, fragment)
        program.bind(self.vbo)
        program["u_antialias"] = u_antialias,
        program["u_model"] = self.model
        program["u_view"] = self.view
        program["u_projection"] = self.projection
        self.program = program


    def on_initialize(self, event):
        gloo.set_state(depth_test=False, blend=True, clear_color=(1, 1, 1, 1),
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')

if __name__ == '__main__':
    canvas = Canvas()
    canvas.show()
    app.run()
