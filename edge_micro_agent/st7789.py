"""
【ファイル概要】
st7789.py

RP2350-GEEKのオンボードLCDをSPI制御するための軽量ディスプレイドライバ。
"""

# Lightweight ST7789 SPI Driver for MicroPython (RP2350-GEEK)

import time
try:
    from machine import Pin, SPI
except ImportError:
    # PC Mock
    class Pin:
        OUT = 0
        def __init__(self, *args, **kwargs): pass
        def value(self, v=None): pass
    class SPI:
        def __init__(self, *args, **kwargs): pass
        def write(self, buf): pass

class ST7789:
    def __init__(self, spi, cs, dc, rst, width=240, height=135, rotation=0):
        self.spi = spi
        self.cs = cs
        self.dc = dc
        self.rst = rst
        self.width = width
        self.height = height
        self.rotation = rotation
        
        self.cs.value(1)
        self.dc.value(0)
        self.reset()
        self.init_display()

    def reset(self):
        self.rst.value(1)
        time.sleep(0.05)
        self.rst.value(0)
        time.sleep(0.05)
        self.rst.value(1)
        time.sleep(0.05)

    def write_cmd(self, cmd):
        self.dc.value(0)
        self.cs.value(0)
        self.spi.write(bytearray([cmd]))
        self.cs.value(1)

    def write_data(self, data):
        self.dc.value(1)
        self.cs.value(0)
        self.spi.write(data if isinstance(data, (bytes, bytearray)) else bytearray([data]))
        self.cs.value(1)

    def init_display(self):
        self.write_cmd(0x11) # Sleep out
        time.sleep(0.12)
        
        # MADCTL (Rotation)
        self.write_cmd(0x36)
        if self.rotation == 0:
            self.write_data(0x00)
        elif self.rotation == 1:
            self.write_data(0x60)
            
        # Color mode (16-bit)
        self.write_cmd(0x3A)
        self.write_data(0x55)
        
        self.write_cmd(0x29) # Display on
        
    def fill_rect(self, x, y, w, h, color):
        # Extremely simplified for mock
        pass
        
    def fill(self, color):
        pass
        
    def text(self, text, x, y, color):
        # Real MicroPython uses framebuf, we just mock it
        print(f"[LCD MOCK] Displaying at ({x},{y}): {text} (Color: {color})")
