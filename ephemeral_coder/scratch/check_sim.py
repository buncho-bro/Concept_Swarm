from sentence_transformers import SentenceTransformer
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')
pairs = [
    ('Quantum Qubit Gate Operation', 'Superconducting Qubit'),
    ('CRISPR Gene Knockout Protocol', 'CRISPR-Cas9 Gene Editing'),
    ('Deep Transformer Neural Network', 'Transformer Attention Mechanism'),
    ('Algorithmic Automated High-Frequency Trading', 'Algorithmic Trading'),
    ('Antibody Drug Conjugate', 'Monoclonal Antibody Therapy'),
    ('Sourdough Bread Fermentation', 'Sourdough Wild Yeast Microbiome'),
    ('Ceramic Pottery Kiln Firing', 'Artisanal Ceramic Glaze'),
    ('Gravitational Wave Laser Interferometer', 'Gravitational Wave Interferometry'),
    ('Carbon Fiber Composite Manufacturing', 'Carbon Fiber Reinforced Polymer')
]

for p1, p2 in pairs:
    e1 = embedder.encode(p1, convert_to_tensor=True)
    e2 = embedder.encode(p2, convert_to_tensor=True)
    sim = torch.dot(e1/torch.norm(e1), e2/torch.norm(e2)).item()
    print(f'{sim:.4f} | "{p1}" <=> "{p2}"')
