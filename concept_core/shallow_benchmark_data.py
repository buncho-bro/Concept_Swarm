"""
shallow_benchmark_data.py -- 浅層抽出ベンチマーク用データセット
================================================================
* 8 ドメイン x アンカー 3 文 (KNN / 物理収束の参照点)
* 単一ドメイン テスト文 48 (各ドメイン 6 文、各文にニッチ専門用語を 1 つ含む)
* 複数ドメイン混在テスト文 16 (正解ドメイン 2 つ)
すべて英語。アンカー文とテスト文で語の重複を極力避けている。
"""

DOMAINS = ["programming", "biology", "finance", "astronomy", "music", "cooking", "law", "medicine"]

ANCHORS = {
    "programming": [
        "Software development: writing functions, classes and loops in a programming language.",
        "Debugging source code, compiling programs and managing software dependencies.",
        "Algorithms and data structures such as arrays, hash tables and recursion.",
    ],
    "biology": [
        "Molecular biology studies DNA, genes, proteins and cells.",
        "Genetics: chromosomes, mutations, heredity and gene expression.",
        "Living organisms, enzymes and the biochemistry of the cell.",
    ],
    "finance": [
        "Finance: banks, loans, interest rates and investment portfolios.",
        "The stock market, bonds, dividends and trading of securities.",
        "Accounting, credit, debt and the management of money.",
    ],
    "astronomy": [
        "Astronomy: stars, planets, galaxies and the universe.",
        "Telescopes observe distant nebulae, comets and black holes.",
        "The orbit of planets around the sun and the expansion of the cosmos.",
    ],
    "music": [
        "Music: melody, harmony, rhythm and musical instruments.",
        "Composers write symphonies and songs using chords and scales.",
        "An orchestra performs a concert with violins, pianos and drums.",
    ],
    "cooking": [
        "Cooking: recipes, ingredients, kitchens and baking food.",
        "Chefs chop vegetables, season meat and simmer sauces.",
        "Frying, roasting and boiling dishes for a delicious meal.",
    ],
    "law": [
        "Law: courts, judges, lawyers and legal contracts.",
        "Criminal trials, evidence, verdicts and the constitution.",
        "Legislation, lawsuits, rights and the justice system.",
    ],
    "medicine": [
        "Medicine: doctors, patients, hospitals and treatment of disease.",
        "Diagnosis of symptoms, prescriptions and surgery.",
        "Clinical care for infections, the heart and the immune system.",
    ],
}

# (text, domain, niche_term)
SINGLE = [
    ("The Kubernetes operator reconciles pod replicas after each rollout.", "programming", "Kubernetes"),
    ("Use memoization to avoid recomputing overlapping subproblems.", "programming", "memoization"),
    ("A segfault occurred because the pointer was dereferenced after free.", "programming", "segfault"),
    ("The coroutine yields control back to the asyncio event loop.", "programming", "coroutine"),
    ("Refactor the monolith into microservices behind a gRPC gateway.", "programming", "gRPC"),
    ("The linter flagged an unused import in the TypeScript module.", "programming", "TypeScript"),

    ("CRISPR guides the Cas9 nuclease to cut a specific genomic locus.", "biology", "CRISPR"),
    ("Reverse transcriptase converts viral RNA into complementary DNA.", "biology", "transcriptase"),
    ("Mitochondria generate ATP through oxidative phosphorylation.", "biology", "phosphorylation"),
    ("Ribosomes translate messenger RNA codons into a polypeptide chain.", "biology", "Ribosomes"),
    ("Epigenetic methylation silences the promoter of the tumor suppressor.", "biology", "methylation"),
    ("Plasmids carrying the antibiotic resistance gene were transformed into E. coli.", "biology", "Plasmids"),

    ("The collateralized debt obligation was split into senior and junior tranches.", "finance", "tranches"),
    ("Quantitative easing pushed bond yields lower across the curve.", "finance", "Quantitative"),
    ("The hedge fund shorted the stock and bought put options as protection.", "finance", "hedge"),
    ("Amortization spreads the mortgage principal over thirty years of payments.", "finance", "Amortization"),
    ("The central bank raised the federal funds rate to curb inflation.", "finance", "inflation"),
    ("Arbitrage traders exploited the spread between futures and spot prices.", "finance", "Arbitrage"),

    ("Stellar parallax lets astronomers measure the distance to nearby stars.", "astronomy", "parallax"),
    ("The redshift of the quasar implies it is billions of light years away.", "astronomy", "redshift"),
    ("A pulsar emits beams of radiation as the neutron star spins.", "astronomy", "pulsar"),
    ("The exoplanet transits its host star every four days.", "astronomy", "exoplanet"),
    ("Gravitational lensing by the cluster bends light from background galaxies.", "astronomy", "lensing"),
    ("The supernova remnant expands into the interstellar medium.", "astronomy", "supernova"),

    ("The pianist played a rapid arpeggio in the final cadenza.", "music", "arpeggio"),
    ("Bach's fugues are masterpieces of counterpoint.", "music", "counterpoint"),
    ("The guitarist tuned down to drop D for heavier riffs.", "music", "riffs"),
    ("The soprano sang the aria with perfect vibrato.", "music", "vibrato"),
    ("The jazz quartet improvised over a twelve-bar blues progression.", "music", "improvised"),
    ("The conductor slowed the tempo during the crescendo.", "music", "crescendo"),

    ("Cook the steak sous-vide at fifty-four degrees, then sear it.", "cooking", "sous"),
    ("Whisk the yolks slowly to emulsify the hollandaise.", "cooking", "hollandaise"),
    ("Caramelize the onions over low heat for forty minutes.", "cooking", "Caramelize"),
    ("Knead the sourdough and let it proof overnight.", "cooking", "sourdough"),
    ("Deglaze the pan with wine to lift the browned fond.", "cooking", "Deglaze"),
    ("Blanch the asparagus, then plunge it into an ice bath.", "cooking", "Blanch"),

    ("The defendant filed a writ of habeas corpus challenging his detention.", "law", "habeas"),
    ("The plaintiff sued for negligence under tort law.", "law", "tort"),
    ("The appellate court overturned the verdict on procedural grounds.", "law", "appellate"),
    ("The witness gave a sworn affidavit before the notary.", "law", "affidavit"),
    ("The judge issued an injunction halting the merger.", "law", "injunction"),
    ("Under the doctrine of stare decisis, precedent binds lower courts.", "law", "precedent"),

    ("The patient developed tachycardia after the anesthesia.", "medicine", "tachycardia"),
    ("Warfarin is an anticoagulant that prevents blood clots.", "medicine", "anticoagulant"),
    ("An MRI revealed a herniated disc compressing the nerve.", "medicine", "herniated"),
    ("The oncologist recommended chemotherapy after the biopsy.", "medicine", "chemotherapy"),
    ("Insulin resistance is a hallmark of type 2 diabetes.", "medicine", "Insulin"),
    ("The vaccine triggers antibodies against the viral antigen.", "medicine", "antibodies"),
]

# (text, [domain_a, domain_b])
MULTI = [
    ("We trained a neural network in Python to predict CRISPR off-target cuts in the genome.", ["programming", "biology"]),
    ("The trading bot written in Rust executes arbitrage between crypto exchanges.", ["programming", "finance"]),
    ("The telescope's image pipeline uses a CUDA kernel to stack exposures of the nebula.", ["programming", "astronomy"]),
    ("The app's audio engine computes a Fourier transform to detect the chord progression.", ["programming", "music"]),
    ("The pharmaceutical company's stock surged after the vaccine trial succeeded.", ["finance", "medicine"]),
    ("The bankruptcy court approved the restructuring of the company's bonds.", ["law", "finance"]),
    ("The surgeon was sued for malpractice after the botched operation.", ["law", "medicine"]),
    ("The chef baked a cake shaped like Saturn with rings of sugar.", ["cooking", "astronomy"]),
    ("The orchestra performed a symphony inspired by the planets and distant stars.", ["music", "astronomy"]),
    ("Fermentation by yeast enzymes makes the sourdough rise.", ["cooking", "biology"]),
    ("The patient's tumor DNA was sequenced to find mutations in the oncogene.", ["medicine", "biology"]),
    ("A jury convicted the hacker of exploiting a buffer overflow in the government's server.", ["law", "programming"]),
    ("The restaurant serves risotto and grilled octopus while a string quartet plays Mozart.", ["music", "cooking"]),
    ("The patent lawsuit claimed the startup copied the gene editing technique.", ["law", "biology"]),
    ("Doctors recommend a diet of steamed vegetables and grilled fish to lower cholesterol.", ["medicine", "cooking"]),
    ("The hedge fund invested heavily in a startup building satellites to map exoplanets.", ["finance", "astronomy"]),
]
