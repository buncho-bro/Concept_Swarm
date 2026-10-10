"""
【ファイル概要】
main.py

Raspberry Pi RP2350 (MicroPython) 上で動作するメインファームウェア。SRAM 520KB以内で局所100語の物理シミュレーションを実行し、推論を行います。
"""

import struct
import sys
import gc
import time

try:
    from ulab import numpy as np
except ImportError:
    import numpy as np

class LocalTopographyLoader:
    def __init__(self, filename="micro_topo.bin"):
        self.f = open(filename, "rb")
        magic = self.f.read(4)
        if magic != b'RPAI':
            raise ValueError("Invalid magic in bin file")
        
        self.n_words, self.dim = struct.unpack('<HH', self.f.read(4))
        
        self.vocab_offset = 8
        self.embed_offset = self.vocab_offset + self.n_words * 16
        self.neigh_offset = self.embed_offset + self.n_words * self.dim * 4
        self.yes_offset = self.neigh_offset + self.n_words * 20 * 2
        
        # Load vocab into RAM (takes ~16KB for 1000 words)
        self.f.seek(self.vocab_offset)
        vocab_bytes = self.f.read(self.n_words * 16)
        self.vocab_dict = {}
        for i in range(self.n_words):
            word_b = vocab_bytes[i*16:(i+1)*16]
            word = word_b.split(b'\x00')[0].decode('ascii')
            self.vocab_dict[word] = i
            
        # Pre-allocate SRAM buffer for local 100 words
        self.local_max = 100
        self.local_embeddings = np.zeros((self.local_max, self.dim), dtype=np.float32)
        self.local_count = 0
        
        # Load Yes/No vectors
        self.f.seek(self.yes_offset)
        self.yes_vec = np.frombuffer(self.f.read(self.dim * 4), dtype=np.float32).copy()
        self.no_vec = np.frombuffer(self.f.read(self.dim * 4), dtype=np.float32).copy()
        
    def load_local_scope(self, text):
        words = text.lower().replace(".", "").replace(",", "").split()
        target_indices = []
        seen = set()
        
        for w in words:
            if w in self.vocab_dict:
                idx = self.vocab_dict[w]
                if idx not in seen:
                    seen.add(idx)
                    target_indices.append(idx)
                    
                # Read neighbors
                self.f.seek(self.neigh_offset + idx * 20 * 2)
                neighbors = struct.unpack('<' + 'H'*20, self.f.read(40))
                for n in neighbors:
                    if n not in seen:
                        seen.add(n)
                        target_indices.append(n)
                        
        # Limit to local_max to prevent SRAM overflow
        target_indices = target_indices[:self.local_max]
        self.local_count = len(target_indices)
        
        # Load into the pre-allocated static matrix
        for i, idx in enumerate(target_indices):
            self.f.seek(self.embed_offset + idx * self.dim * 4)
            buf = self.f.read(self.dim * 4)
            self.local_embeddings[i] = np.frombuffer(buf, dtype=np.float32)
            
        return self.local_count


class MicroPhysicsEngine:
    def __init__(self, loader):
        self.loader = loader
        
    def run_simulation(self, iterations=5, sigma=0.2):
        if self.loader.local_count == 0:
            return np.zeros(self.loader.dim, dtype=np.float32)
            
        # Get active slice of the pre-allocated matrix (Zero Copy View)
        V = self.loader.local_embeddings[:self.loader.local_count]
        
        # Initial point: Mean of the input tokens (we just use mean of all loaded for simplicity)
        # In ulab, sum over axis=0 might not be supported natively in all versions, 
        # so we can use a loop or np.dot with ones
        ones = np.ones(self.loader.local_count, dtype=np.float32)
        x = np.dot(ones, V)
        norm_x = np.sqrt(np.dot(x, x))
        x = x / max(norm_x, 1e-9)
        
        # Gradient Ascent for Interference Resonance
        for _ in range(iterations):
            # Compute Cosine Similarities: V * x (both are unit vectors)
            sims = np.dot(V, x)
            # Squared distance = 2 - 2 * sim
            sq_dist = 2.0 - 2.0 * sims
            # Gaussian weights
            weights = np.exp(-sq_dist / (2.0 * sigma * sigma))
            # New direction is weighted sum of V
            grad = np.dot(weights, V)
            
            # Step
            x = x + 0.5 * grad
            norm_x = np.sqrt(np.dot(x, x))
            x = x / max(norm_x, 1e-9)
            
        return x

class GeekDisplay:
    def __init__(self):
        try:
            from machine import Pin, SPI
            self.is_mock = False
        except ImportError:
            # Running on PC
            self.is_mock = True
            
        import st7789
        
        # RP2350-GEEK typical pins
        if not self.is_mock:
            from machine import Pin, SPI
            spi = SPI(1, baudrate=30000000, polarity=1, phase=1, sck=Pin(10), mosi=Pin(11))
            cs = Pin(9, Pin.OUT)
            dc = Pin(8, Pin.OUT)
            rst = Pin(12, Pin.OUT)
        else:
            # Mock objects handled in st7789.py
            spi = st7789.SPI()
            cs = st7789.Pin()
            dc = st7789.Pin()
            rst = st7789.Pin()
            
        self.lcd = st7789.ST7789(spi, cs, dc, rst, width=240, height=135, rotation=1)
        self.lcd.fill(0x0000) # Black
        
    def show_result(self, decision, conf, latency):
        color = 0x07E0 if decision == "YES" else 0xF800 # Green vs Red
        self.lcd.fill(0x0000)
        self.lcd.text(decision, 80, 50, color)
        self.lcd.text(f"Conf: {conf:.2f}", 60, 80, 0xFFFF)
        self.lcd.text(f"Lat: {latency:.1f}ms", 60, 100, 0xFFFF)


def main():
    gc.collect()
    start_ram = gc.mem_free() if hasattr(gc, 'mem_free') else 0
    print(f"--- RP2350-GEEK Micro Agent Booting ---")
    print(f"Free SRAM: {start_ram / 1024:.1f} KB" if start_ram else "Running on PC (Mock Mode)")
    
    loader = LocalTopographyLoader("micro_topo.bin")
    engine = MicroPhysicsEngine(loader)
    display = GeekDisplay()
    
    print("Ready. Type a sentence and press Enter.")
    
    test_cases = [
        "is this a very good amazing excellent idea",
        "this is a terrible awful worst idea ever",
        "i agree yes very good",
        "i disagree no bad terrible"
    ]
    
    for text in test_cases:
        t0 = time.time()
        
        # 1. Load Local Scope (Max 100 words from Flash to SRAM)
        count = loader.load_local_scope(text)
        
        # 2. Physics Simulation
        x_star = engine.run_simulation(iterations=5, sigma=0.3)
        
        # 3. Decision
        sim_yes = np.dot(x_star, loader.yes_vec)
        sim_no = np.dot(x_star, loader.no_vec)
        
        t1 = time.time()
        latency = (t1 - t0) * 1000
        
        decision = "YES" if sim_yes > sim_no else "NO"
        conf = abs(sim_yes - sim_no)
        
        print(f"\nInput: '{text}'")
        print(f"Loaded {count} local vectors.")
        print(f"Sim_Yes: {sim_yes:.3f} | Sim_No: {sim_no:.3f}")
        print(f"Decision: [{decision}] (Conf: {conf:.3f})")
        print(f"Latency: {latency:.1f} ms")
        
        # 4. Display on LCD
        display.show_result(decision, conf, latency)
        
        if hasattr(gc, 'mem_free'):
            print(f"Free SRAM remaining: {gc.mem_free() / 1024:.1f} KB")

if __name__ == "__main__":
    main()
