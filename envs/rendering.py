
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        

try:
    from gym import error
except ImportError:
    from gymnasium import error

try:
    import pyglet
except ImportError as e:
    print(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    print(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np
import os
from datetime import datetime

RAD2DEG = 57.29577951308232

def _generate_trajectory_filename(prefix="trajectory", is_animated=False):
    
    
    try:
        from onpolicy.envs.uavs import settings
        algorithm = getattr(settings, 'algorithm', 'unknown')
    except:
        algorithm = 'unknown'
    
    
    trajectory_dir = "trajectory"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    
    
    if is_animated:
        filename = f"{prefix}_{algorithm}.gif"
    else:
        filename = f"{prefix}_{algorithm}.png"
    
    return os.path.join(trajectory_dir, filename)

def _sample_trajectory_for_display(trajectory, sample_step=1):
    
    if not trajectory or sample_step <= 1:
        return trajectory
    
    sampled = []
    sampled_indices = set()
    
    
    sampled.append(trajectory[0])
    sampled_indices.add(0)
    
    
    last_sampled_idx = 0
    for i in range(sample_step, len(trajectory), sample_step):
        if i < len(trajectory):
            
            prev_pos = trajectory[last_sampled_idx]
            curr_pos = trajectory[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            
            
            if distance > 3.0:  
                
                mid_indices = list(range(last_sampled_idx + 1, i))
                for mid_idx in mid_indices[::max(1, len(mid_indices)//3)]:  
                    if mid_idx not in sampled_indices:
                        sampled.append(trajectory[mid_idx])
                        sampled_indices.add(mid_idx)
            
            sampled.append(trajectory[i])
            sampled_indices.add(i)
            last_sampled_idx = i
    
    
    last_index = len(trajectory) - 1
    if len(trajectory) > 1 and last_index not in sampled_indices:
        
        if sampled:
            last_sampled_pos = sampled[-1]
            last_pos = trajectory[-1]
            distance = np.sqrt((last_pos[0] - last_sampled_pos[0])**2 + (last_pos[1] - last_sampled_pos[1])**2)
            
            
            if distance > 3.0:
                start_idx = max(0, last_index - sample_step + 1)
                mid_indices = list(range(start_idx, last_index))
                for mid_idx in mid_indices[::max(1, len(mid_indices)//2)]:
                    if mid_idx not in sampled_indices and mid_idx < len(trajectory):
                        sampled.append(trajectory[mid_idx])
                        sampled_indices.add(mid_idx)
        
        sampled.append(trajectory[-1])
    
    return sampled

def smooth_trajectory(x_coords, y_coords, smooth_method='spline', smooth_level=0.15, interpolation_factor=2, corner_threshold=0.5):
    
    if len(x_coords) < 3 or smooth_method == 'none' or smooth_level <= 0:
        return x_coords, y_coords
    
    try:
        
        important_indices = [0]  
        
        if len(x_coords) > 2 and corner_threshold > 0:
            for i in range(1, len(x_coords) - 1):
                
                v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
                v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
                
                
                if np.linalg.norm(v1) > 0.001 and np.linalg.norm(v2) > 0.001:
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle)
                    
                    
                    if angle_change > corner_threshold:
                        important_indices.append(i)
        
        important_indices.append(len(x_coords) - 1)  
        important_indices = sorted(list(set(important_indices)))
        
        if smooth_method == 'spline' and len(x_coords) >= 4:
            from scipy.interpolate import make_interp_spline
            
            
            num_points = max(len(x_coords), int(len(x_coords) * interpolation_factor))
            
            
            t = np.linspace(0, 1, len(x_coords))
            t_smooth = np.linspace(0, 1, num_points)
            
            
            spline_x = make_interp_spline(t, x_coords, k=min(3, len(x_coords)-1))
            spline_y = make_interp_spline(t, y_coords, k=min(3, len(x_coords)-1))
            x_interp = spline_x(t_smooth)
            y_interp = spline_y(t_smooth)
            
            
            if smooth_level < 1.0:
                
                x_final = []
                y_final = []
                
                for i, (x_orig, y_orig) in enumerate(zip(x_coords, y_coords)):
                    
                    interp_idx = int(i * (len(x_interp) - 1) / (len(x_coords) - 1))
                    x_interp_val = x_interp[interp_idx]
                    y_interp_val = y_interp[interp_idx]
                    
                    
                    preservation_factor = 0.0
                    if i in important_indices:
                        preservation_factor = 1.0 - smooth_level
                    
                    
                    mix_factor = smooth_level * (1.0 - preservation_factor)
                    x_mixed = x_orig * (1 - mix_factor) + x_interp_val * mix_factor
                    y_mixed = y_orig * (1 - mix_factor) + y_interp_val * mix_factor
                    
                    x_final.append(x_mixed)
                    y_final.append(y_mixed)
                
                return x_final, y_final
            else:
                return x_interp.tolist(), y_interp.tolist()
        
        elif smooth_method == 'linear':
            
            num_points = int(len(x_coords) * interpolation_factor)
            t = np.linspace(0, 1, len(x_coords))
            t_smooth = np.linspace(0, 1, num_points)
            
            x_interp = np.interp(t_smooth, t, x_coords)
            y_interp = np.interp(t_smooth, t, y_coords)
            
            return x_interp.tolist(), y_interp.tolist()
        
        else:
            return x_coords, y_coords
            
    except ImportError:
        
        if len(x_coords) < 3:
            return x_coords, y_coords
        
        window_size = max(2, int(3 * smooth_level))
        x_smooth = []
        y_smooth = []
        
        for i in range(len(x_coords)):
            if i in important_indices:
                
                x_smooth.append(x_coords[i])
                y_smooth.append(y_coords[i])
            else:
                
                start = max(0, i - window_size // 2)
                end = min(len(x_coords), i + window_size // 2 + 1)
                
                x_avg = sum(x_coords[start:end]) / (end - start)
                y_avg = sum(y_coords[start:end]) / (end - start)
                
                
                x_mixed = x_coords[i] * (1 - smooth_level) + x_avg * smooth_level
                y_mixed = y_coords[i] * (1 - smooth_level) + y_avg * smooth_level
                
                x_smooth.append(x_mixed)
                y_smooth.append(y_mixed)
        
        return x_smooth, y_smooth
    
    except Exception:
        
        return x_coords, y_coords

def get_display(spec):
    
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        
        glEnable(GL_LINE_SMOOTH)
        
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
            
            
            
            
            
            
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) 
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) 
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  
        glEnd()

        color = (self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  
        glEnd()

class TextGeom(Geom):
    def __init__(self, text, font_size=14, x=0, y=0):
        super().__init__()
        self.text = text
        self.font_size = font_size
        self.x = x
        self.y = y
        self.label = pyglet.text.Label(text,
                                       font_name='Times New Roman',
                                       font_size=font_size,
                                       x=x, y=y,
                                       anchor_x='center', anchor_y='center',
                                       color=(255, 255, 255, 255))  

    def render1(self):
        
        
        
        

        
        
        
        

        glPushAttrib(GL_ENABLE_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.label.draw()

        glPopAttrib()

    def set_color(self, r, g, b, alpha=1):
        super().set_color(r, g, b, alpha)  
        self.label.color = (int(r * 255), int(g * 255), int(b * 255), int(alpha * 255))  


def add_text(viewer, text, x, y, font_size=14):
    text_geom = TextGeom(text, font_size, x, y)
    viewer.add_geom(text_geom)

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)






def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)



class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()


def plot_uav_trajectories(trajectories_data, map_size=(20, 20), save_path=None, max_uavs=None, show_gaussian_centers=True, subplot_ax=None, legend_loc='upper left'):
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("需要安装matplotlib才能绘制轨迹图: pip install matplotlib")
        return
    
    if not trajectories_data:
        print("没有找到轨迹数据")
        return
    
    
    if subplot_ax is not None:
        ax = subplot_ax
    else:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
    
    
    distinctive_colors = [
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
        '
    ]
    colors = distinctive_colors
    
    
    uav_ids = list(trajectories_data.keys())
    if max_uavs is not None and max_uavs > 0:
        uav_ids = sorted(uav_ids)[:max_uavs]
    
    
    trajectory_legends = []
    start_legends = []
    end_legends = []
    
    for i, uav_id in enumerate(uav_ids):
        trajectory = trajectories_data[uav_id]
        if not trajectory:
            continue
            
        
        try:
            from onpolicy.envs.uavs import settings
            use_gradient = getattr(settings, 'static_trajectory_alpha_gradient', True)
            use_smooth = getattr(settings, 'static_trajectory_smooth', True)
            use_thin_lines = getattr(settings, 'static_trajectory_thin_lines', True)
            show_direction = getattr(settings, 'static_trajectory_show_direction', True)
            
            
            sample_step = getattr(settings, 'trajectory_sample_step', 1)
            smooth_enabled = getattr(settings, 'trajectory_smooth_enabled', True)
            smooth_level = getattr(settings, 'trajectory_smooth_level', 0.15)
            smooth_method = getattr(settings, 'trajectory_smooth_method', 'spline')
            interpolation_factor = getattr(settings, 'trajectory_interpolation_factor', 2)
            corner_threshold = getattr(settings, 'trajectory_corner_threshold', 0.5)
        except:
            use_gradient = True
            use_smooth = True
            use_thin_lines = True
            show_direction = True
            sample_step = 1
            smooth_enabled = True
            smooth_level = 0.15
            smooth_method = 'spline'
            interpolation_factor = 2
            corner_threshold = 0.5
        
        
        sampled_trajectory = _sample_trajectory_for_display(trajectory, sample_step)
        
        
        try:
            from onpolicy.envs.uavs import settings
            map_length = getattr(settings, 'length', 20)
            map_width = getattr(settings, 'width', 20)
        except:
            map_length, map_width = 20, 20  
        
        
        scale_x = 1000.0 / map_length
        scale_y = 1000.0 / map_width
        
        
        def clamp_coordinate(coord, max_val=1000):
            
            return max(0, min(coord, max_val))
        
        x_coords = [clamp_coordinate(pos[0] * scale_x) for pos in sampled_trajectory]
        y_coords = [clamp_coordinate(pos[1] * scale_y) for pos in sampled_trajectory]
        steps = [pos[2] for pos in sampled_trajectory]
        
        
        try:
            if isinstance(uav_id, str):
                uav_index = int(uav_id.split('-')[1]) if '-' in uav_id else int(uav_id)
            else:
                uav_index = int(uav_id)
        except (ValueError, IndexError):
            uav_index = i  
        
        color = colors[uav_index % len(colors)]
        
        
        linewidth = 1.0 if use_thin_lines else 2.0
        
        
        if use_smooth and smooth_enabled and len(x_coords) > 2:
            x_smooth, y_smooth = smooth_trajectory(
                x_coords, y_coords,
                smooth_method=smooth_method,
                smooth_level=smooth_level,
                interpolation_factor=interpolation_factor,
                corner_threshold=corner_threshold
            )
            
            x_smooth = [clamp_coordinate(x) for x in x_smooth]
            y_smooth = [clamp_coordinate(y) for y in y_smooth]
            
            
            if len(x_smooth) > 0 and len(x_coords) > 0:
                x_smooth[0] = x_coords[0]
                y_smooth[0] = y_coords[0]
        else:
            x_smooth, y_smooth = x_coords, y_coords
        
        
        if use_gradient and len(x_smooth) > 1:
            
            n_segments = min(len(x_smooth) - 1, 20)  
            segment_size = max(1, (len(x_smooth) - 1) // n_segments)
            
            for seg in range(n_segments):
                start_idx = seg * segment_size
                
                if seg < n_segments - 1:
                    end_idx = min((seg + 1) * segment_size + 1, len(x_smooth))
                else:
                    
                    end_idx = len(x_smooth)
                
                if start_idx >= len(x_smooth) - 1:
                    break
                
                
                alpha = 0.3 + 0.5 * (seg / max(1, n_segments - 1))
                
                seg_x = x_smooth[start_idx:end_idx]
                seg_y = y_smooth[start_idx:end_idx]
                
                if len(seg_x) > 1:
                    line = ax.plot(seg_x, seg_y, color=color, alpha=alpha, 
                                   linewidth=linewidth, solid_capstyle='round')[0]
                    
                    
                    if seg == 0 and i == 0:
                        
                        legend_line, = ax.plot([], [], color='black', alpha=0.7, linewidth=linewidth)
                        trajectory_legends.append((legend_line, 'Trajectory'))
                
                
                if seg < n_segments - 1 and end_idx < len(x_smooth):
                    next_start_idx = (seg + 1) * segment_size
                    if next_start_idx < len(x_smooth) and end_idx - 1 < len(x_smooth):
                        
                        curr_end_x, curr_end_y = x_smooth[end_idx - 1], y_smooth[end_idx - 1]
                        next_start_x, next_start_y = x_smooth[next_start_idx], y_smooth[next_start_idx]
                        
                        
                        if abs(curr_end_x - next_start_x) > 0.01 or abs(curr_end_y - next_start_y) > 0.01:
                            ax.plot([curr_end_x, next_start_x], [curr_end_y, next_start_y], 
                                   color=color, alpha=alpha, linewidth=linewidth, solid_capstyle='round')
        else:
            
            line, = ax.plot(x_smooth, y_smooth, color=color, alpha=0.7, linewidth=linewidth)
            
            if i == 0:
                
                legend_line, = ax.plot([], [], color='black', alpha=0.7, linewidth=linewidth)
                trajectory_legends.append((legend_line, 'Trajectory'))
        
        
        if show_direction and len(x_coords) > 1:
            
            arrow_positions = [len(x_coords) // 3, 2 * len(x_coords) // 3]
            for pos in arrow_positions:
                if pos < len(x_coords) - 1:
                    dx = x_coords[pos + 1] - x_coords[pos]
                    dy = y_coords[pos + 1] - y_coords[pos]
                    
                    
                    if abs(dx) + abs(dy) > 0.1:
                        ax.annotate('', xy=(x_coords[pos + 1], y_coords[pos + 1]),
                                   xytext=(x_coords[pos], y_coords[pos]),
                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                 alpha=0.6, lw=1.0))
        
        
        if len(x_smooth) > 0:
            
            start_x, start_y = x_smooth[0], y_smooth[0]
            end_x, end_y = x_smooth[-1], y_smooth[-1]
            
            
            if i == 0:  
                
                legend_start = ax.scatter([], [], color='black', s=80, marker='o', 
                                        edgecolors='white', linewidth=2, alpha=0.9)
                start_legends.append((legend_start, 'Start Point'))
            
            if i == 0:  
                
                legend_end = ax.scatter([], [], color='black', s=120, marker='s', 
                                      edgecolors='white', linewidth=2, alpha=1.0)
                end_legends.append((legend_end, 'End Point'))
            
            
            def find_optimal_icon_position(x, y, existing_icons, icon_size=60):
                
                
                original_conflict = False
                for existing_x, existing_y, existing_size in existing_icons:
                    distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
                    min_distance = (icon_size + existing_size) / 2 + 5  
                    if distance < min_distance:
                        original_conflict = True
                        break
                
                
                if not original_conflict:
                    return 0, 0
                
                
                candidates = [
                    (icon_size//2, icon_size//2),      
                    (-icon_size//2, icon_size//2),     
                    (icon_size//2, -icon_size//2),     
                    (-icon_size//2, -icon_size//2),    
                    (0, icon_size//2),                  
                    (0, -icon_size//2),                
                    (icon_size//2, 0),                 
                    (-icon_size//2, 0),                
                ]
                
                
                for multiplier in range(2, 4):  
                    extended_candidates = [
                        (icon_size//2 * multiplier, icon_size//2 * multiplier),
                        (-icon_size//2 * multiplier, icon_size//2 * multiplier),
                        (icon_size//2 * multiplier, -icon_size//2 * multiplier),
                        (-icon_size//2 * multiplier, -icon_size//2 * multiplier),
                        (0, icon_size//2 * multiplier),
                        (0, -icon_size//2 * multiplier),
                        (icon_size//2 * multiplier, 0),
                        (-icon_size//2 * multiplier, 0),
                    ]
                    candidates.extend(extended_candidates)
                
                
                for offset_x, offset_y in candidates:
                    new_x, new_y = x + offset_x, y + offset_y
                    conflict = False
                    
                    
                    for existing_x, existing_y, existing_size in existing_icons:
                        distance = ((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2) ** 0.5
                        min_distance = (icon_size + existing_size) / 2 + 5  
                        if distance < min_distance:
                            conflict = True
                            break
                    
                    if not conflict:
                        return offset_x, offset_y
                
                
                return icon_size//2, icon_size//2
            
            
            existing_icons = []
            
            
            for j in range(i):  
                if j < len(uav_ids):
                    prev_uav_id = uav_ids[j]
                    prev_trajectory = trajectories_data[prev_uav_id]
                    if prev_trajectory:
                        
                        prev_sampled = _sample_trajectory_for_display(prev_trajectory, sample_step)
                        prev_x_coords = [clamp_coordinate(pos[0] * scale_x) for pos in prev_sampled]
                        prev_y_coords = [clamp_coordinate(pos[1] * scale_y) for pos in prev_sampled]
                        
                        if prev_x_coords and prev_y_coords:
                            
                            if use_smooth and smooth_enabled and len(prev_x_coords) > 2:
                                prev_x_smooth, prev_y_smooth = smooth_trajectory(
                                    prev_x_coords, prev_y_coords,
                                    smooth_method=smooth_method,
                                    smooth_level=smooth_level,
                                    interpolation_factor=interpolation_factor,
                                    corner_threshold=corner_threshold
                                )
                                prev_x_smooth = [clamp_coordinate(x) for x in prev_x_smooth]
                                prev_y_smooth = [clamp_coordinate(y) for y in prev_y_smooth]
                            else:
                                prev_x_smooth, prev_y_smooth = prev_x_coords, prev_y_coords
                            
                            
                            existing_icons.extend([
                                (prev_x_smooth[0], prev_y_smooth[0], 40),  
                                (prev_x_smooth[-1], prev_y_smooth[-1], 60)  
                            ])
            
            
            if show_gaussian_centers:
                try:
                    from onpolicy.envs.uavs import settings
                    if hasattr(settings, 'use_gaussian_task_generation') and settings.use_gaussian_task_generation:
                        centers = settings.gaussian_centers
                        for center_x, center_y in centers:
                            
                            try:
                                from onpolicy.envs.uavs import settings
                                map_length = getattr(settings, 'length', 20)
                                map_width = getattr(settings, 'width', 20)
                            except:
                                map_length, map_width = 20, 20
                            
                            scale_x_gaussian = 1000.0 / map_length
                            scale_y_gaussian = 1000.0 / map_width
                            
                            
                            def clamp_coordinate(coord, max_val=1000):
                                return max(0, min(coord, max_val))
                            
                            scaled_center_x = clamp_coordinate(center_x * scale_x_gaussian)
                            scaled_center_y = clamp_coordinate(center_y * scale_y_gaussian)
                            
                            
                            existing_icons.append((scaled_center_x, scaled_center_y, 100))
                except:
                    pass  
            
            
            start_offset_x, start_offset_y = find_optimal_icon_position(
                start_x, start_y, existing_icons, icon_size=40)
            
            
            end_offset_x, end_offset_y = find_optimal_icon_position(
                end_x, end_y, existing_icons, icon_size=60)
            
            
            if start_offset_x != 0 or start_offset_y != 0:
                
                start_point = ax.scatter(start_x + start_offset_x, start_y + start_offset_y, 
                                       color=color, s=80, marker='o', edgecolors='white', 
                                       linewidth=2, zorder=5, alpha=0.9)
                
                ax.plot([start_x, start_x + start_offset_x], 
                       [start_y, start_y + start_offset_y], 
                       color=color, linewidth=2, alpha=0.8)
            else:
                
                start_point = ax.scatter(start_x, start_y, color=color, s=80, 
                                       marker='o', edgecolors='white', linewidth=2, zorder=5,
                                       alpha=0.9)
            
            if end_offset_x != 0 or end_offset_y != 0:
                
                end_point = ax.scatter(end_x + end_offset_x, end_y + end_offset_y, 
                                     color=color, s=120, marker='s', edgecolors='white', 
                                     linewidth=2, zorder=5, alpha=1.0)
                
                ax.plot([end_x, end_x + end_offset_x], 
                       [end_y, end_y + end_offset_y], 
                       color=color, linewidth=2, alpha=0.8)
            else:
                
                end_point = ax.scatter(end_x, end_y, color=color, s=120, 
                                     marker='s', edgecolors='white', linewidth=2, zorder=5,
                                     alpha=1.0)
    
    
    if show_gaussian_centers:
        try:
            from onpolicy.envs.uavs import settings
            if hasattr(settings, 'use_gaussian_task_generation') and settings.use_gaussian_task_generation:
                centers = settings.gaussian_centers
                sigmas = settings.gaussian_sigmas if hasattr(settings, 'gaussian_sigmas') else [1.0] * len(centers)
                
                
                if not isinstance(sigmas, (list, tuple)):
                    sigmas = [sigmas] * len(centers)
                
                for i, (center_x, center_y) in enumerate(centers):
                    sigma = sigmas[i] if i < len(sigmas) else sigmas[0]
                    
                    
                    try:
                        from onpolicy.envs.uavs import settings
                        map_length = getattr(settings, 'length', 20)
                        map_width = getattr(settings, 'width', 20)
                    except:
                        map_length, map_width = 20, 20  
                    
                    
                    scale_x = 1000.0 / map_length
                    scale_y = 1000.0 / map_width
                    
                    
                    def clamp_coordinate(coord, max_val=1000):
                        
                        return max(0, min(coord, max_val))
                    
                    scaled_center_x = clamp_coordinate(center_x * scale_x)
                    scaled_center_y = clamp_coordinate(center_y * scale_y)
                    
                    scaled_sigma = sigma * (scale_x + scale_y) / 2
                    
                    
                    center_point = ax.scatter(scaled_center_x, scaled_center_y, color='red', s=200, 
                                             marker='*', edgecolors='black', linewidth=3, 
                                             zorder=10)
                    
                    
                    circle1 = plt.Circle((scaled_center_x, scaled_center_y), scaled_sigma, color='red', 
                                       fill=False, linestyle='--', alpha=0.5, linewidth=1)
                    circle2 = plt.Circle((scaled_center_x, scaled_center_y), 2*scaled_sigma, color='red', 
                                       fill=False, linestyle=':', alpha=0.3, linewidth=1)
                    
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)
                    
                    
                    ax.annotate(f'Center{i+1}', 
                               (scaled_center_x, scaled_center_y), 
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=8, ha='left')
                    
                    if i == 0:  
                        
                        legend_center = ax.scatter([], [], color='red', s=200, marker='*', 
                                                 edgecolors='black', linewidth=3)
                        trajectory_legends.append((legend_center, 'Gaussian Centers'))
        except Exception as e:
            print(f"绘制高斯中心点时出错: {e}")
    
    
    ax.set_xlim(-5, 1000)
    ax.set_ylim(-5, 1000)
    ax.set_aspect('equal')  
    
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('
    
    
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)
    
    
    ax.set_xlabel('X Coordinate (m)', fontsize=15)
    ax.set_ylabel('Y Coordinate (m)', fontsize=15)
    
    
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    ax.tick_params(axis='x', which='major', labelbottom=True)
    ax.tick_params(axis='y', which='major', labelleft=True)
    
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    
    
    all_handles = [item[0] for item in start_legends + end_legends + trajectory_legends]
    all_labels = [item[1] for item in start_legends + end_legends + trajectory_legends]
    
    
    if all_handles:  
        print(f"DEBUG: 创建图例，位置: {legend_loc}")  
        
        
        if subplot_ax is not None:
            
            legend_fontsize = 11
            legend_title_fontsize = 12
        else:
            
            legend_fontsize = 13
            legend_title_fontsize = 14
        
        legend = ax.legend(all_handles, all_labels, loc=legend_loc,
                          frameon=True, fancybox=True, shadow=True, ncol=1,
                          fontsize=legend_fontsize, title_fontsize=legend_title_fontsize)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
    
    
    
    
    if subplot_ax is None:
        plt.tight_layout()
        
        if save_path:
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")
        else:
            
            auto_filename = _generate_trajectory_filename("static_trajectory", is_animated=False)
            plt.savefig(auto_filename, dpi=300, bbox_inches='tight')
            print(f"轨迹图已自动保存到: {auto_filename}")
        
        plt.close()

def plot_uav_trajectories_animated(trajectories_data, map_size=(20, 20), save_path=None, max_uavs=None, show_gaussian_centers=True, use_tail_effect=True):
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.colors as mcolors
        import numpy as np
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("需要安装matplotlib才能绘制动画: pip install matplotlib")
        return
    
    if not trajectories_data:
        print("没有找到轨迹数据")
        return
    
    
    uav_ids = list(trajectories_data.keys())
    if max_uavs is not None and max_uavs > 0:
        uav_ids = sorted(uav_ids)[:max_uavs]
    
    
    filtered_data = {uav_id: trajectories_data[uav_id] for uav_id in uav_ids}
    
    
    distinctive_colors = [
        '
        '
        '
    ]
    colors = distinctive_colors
    
    
    all_steps = set()
    for trajectory in filtered_data.values():
        for pos in trajectory:
            all_steps.add(pos[2])
    
    if not all_steps:
        print("没有轨迹数据")
        return
    
    max_step = max(all_steps)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')  
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate (m)', fontsize=15)
        ax.set_ylabel('Y Coordinate (m)', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_yticks([0, 200, 400, 600, 800, 1000])
        
        for i, (uav_id, trajectory) in enumerate(filtered_data.items()):
            color = colors[i % len(colors)]
            
            
            try:
                from onpolicy.envs.uavs import settings
                map_length = getattr(settings, 'length', 20)
                map_width = getattr(settings, 'width', 20)
            except:
                map_length, map_width = 20, 20  
            
            
            scale_x = 1000.0 / map_length
            scale_y = 1000.0 / map_width
            
            
            def clamp_coordinate(coord, max_val=1000):
                
                return max(0, min(coord, max_val))
            
            current_positions = [(clamp_coordinate(pos[0] * scale_x), clamp_coordinate(pos[1] * scale_y)) for pos in trajectory if pos[2] <= frame]
            
            if current_positions:
                
                tail_length = 0
                if use_tail_effect:
                    try:
                        from onpolicy.envs.uavs import settings
                        tail_length = getattr(settings, 'trajectory_tail_length', 3)
                    except:
                        tail_length = 3
                
                
                if use_tail_effect and tail_length > 0 and len(current_positions) > tail_length:
                    
                    display_positions = current_positions[-tail_length:]
                else:
                    
                    display_positions = current_positions
                
                x_coords = [pos[0] for pos in display_positions]
                y_coords = [pos[1] for pos in display_positions]
                
                
                if len(x_coords) > 2:
                    try:
                        from onpolicy.envs.uavs import settings
                        smooth_enabled = getattr(settings, 'trajectory_smooth_enabled', True)
                        smooth_level = getattr(settings, 'trajectory_smooth_level', 0.15)
                        smooth_method = getattr(settings, 'trajectory_smooth_method', 'spline')
                        interpolation_factor = getattr(settings, 'trajectory_interpolation_factor', 2)
                        corner_threshold = getattr(settings, 'trajectory_corner_threshold', 0.5)
                        
                        if smooth_enabled:
                            x_coords, y_coords = smooth_trajectory(
                                x_coords, y_coords,
                                smooth_method=smooth_method,
                                smooth_level=smooth_level,
                                interpolation_factor=interpolation_factor,
                                corner_threshold=corner_threshold
                            )
                            
                            x_coords = [clamp_coordinate(x) for x in x_coords]
                            y_coords = [clamp_coordinate(y) for y in y_coords]
                    except:
                        pass  
                
                
                if len(x_coords) > 1:
                    
                    for j in range(len(x_coords) - 1):
                        alpha = 0.3 + 0.4 * (j + 1) / len(x_coords)  
                        ax.plot(x_coords[j:j+2], y_coords[j:j+2], 
                               color=color, alpha=alpha, linewidth=2)
                
                
                if current_positions:  
                    current_x = current_positions[-1][0]
                    current_y = current_positions[-1][1]
                    ax.scatter(current_x, current_y, color=color, s=150, 
                              marker='o', edgecolors='black', linewidth=2, 
                              label=f'UAV-{uav_id}')
        
        
        if show_gaussian_centers:
            try:
                from onpolicy.envs.uavs import settings
                if hasattr(settings, 'use_gaussian_task_generation') and settings.use_gaussian_task_generation:
                    centers = settings.gaussian_centers
                    sigmas = settings.gaussian_sigmas if hasattr(settings, 'gaussian_sigmas') else [1.0] * len(centers)
                    
                    
                    if not isinstance(sigmas, (list, tuple)):
                        sigmas = [sigmas] * len(centers)
                    
                    for i, (center_x, center_y) in enumerate(centers):
                        sigma = sigmas[i] if i < len(sigmas) else sigmas[0]
                        
                        
                        try:
                            from onpolicy.envs.uavs import settings
                            map_length = getattr(settings, 'length', 20)
                            map_width = getattr(settings, 'width', 20)
                        except:
                            map_length, map_width = 20, 20  
                        
                        
                        scale_x = 1000.0 / map_length
                        scale_y = 1000.0 / map_width
                        
                        
                        def clamp_coordinate(coord, max_val=1000):
                            
                            return max(0, min(coord, max_val))
                        
                        scaled_center_x = clamp_coordinate(center_x * scale_x)
                        scaled_center_y = clamp_coordinate(center_y * scale_y)
                        
                        scaled_sigma = sigma * (scale_x + scale_y) / 2
                        
                        
                        ax.scatter(scaled_center_x, scaled_center_y, color='red', s=200, 
                                 marker='*', edgecolors='black', linewidth=3, 
                                 zorder=10, label='Gaussian Center' if i == 0 else "")
                        
                        
                        circle1 = plt.Circle((scaled_center_x, scaled_center_y), scaled_sigma, color='red', 
                                           fill=False, linestyle='--', alpha=0.5, linewidth=1)
                        circle2 = plt.Circle((scaled_center_x, scaled_center_y), 2*scaled_sigma, color='red', 
                                           fill=False, linestyle=':', alpha=0.3, linewidth=1)
                        
                        ax.add_patch(circle1)
                        ax.add_patch(circle2)
                        
                        
                        ax.annotate(f'C{i+1}', (scaled_center_x, scaled_center_y), 
                                   xytext=(5, 5), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                                   fontsize=8, ha='left')
            except Exception as e:
                pass  
        
        
        
    
    ani = animation.FuncAnimation(fig, animate, frames=range(max_step + 1), 
                                 interval=200, repeat=True)
    
    if save_path:
        
        ani.save(save_path, writer='pillow', fps=5)
        print(f"动画已保存到: {save_path}")
    else:
        
        auto_filename = _generate_trajectory_filename("animated_trajectory", is_animated=True)
        ani.save(auto_filename, writer='pillow', fps=5)
        print(f"动画已自动保存到: {auto_filename}")
    
    plt.close()

def test_trajectory_plotting():
    
    
    import tempfile
    import sys
    import os
    
    
    settings_path = os.path.join(os.path.dirname(__file__))
    if settings_path not in sys.path:
        sys.path.append(settings_path)
    
    try:
        from . import settings
        
        settings.use_gaussian_task_generation = True
        settings.gaussian_centers = [(5, 5), (15, 15)]
        settings.gaussian_sigmas = [2.0, 1.5]
    except:
        pass
    
    
    test_data = {
        0: {  
            0: [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 2, 3), (4, 3, 4)],  
            1: [(5, 5, 0), (4, 4, 1), (3, 3, 2), (2, 4, 3), (1, 5, 4)],  
            2: [(10, 0, 0), (10, 1, 1), (9, 2, 2), (8, 2, 3), (7, 3, 4)]  
        }
    }
    
    print("测试静态轨迹图（包含高斯中心点）...")
    plot_uav_trajectories(test_data, run_id=0, map_size=(20, 20), show_gaussian_centers=True)
    
    print("测试动态轨迹图（包含高斯中心点和拖尾效果）...")
    plot_uav_trajectories_animated(test_data, run_id=0, map_size=(20, 20), 
                                  show_gaussian_centers=True, use_tail_effect=True)
