"""
Minecraft Technical Knowledge Corpus Builder.
Constructs an authoritative, comprehensive domain knowledge dataset covering:
1. Data Packs (mcfunction, modern 1.20.5+/1.21+ Item Components, macros, predicates, tags, worldgen)
2. Resource Packs (blockstates, models, textures, pack.mcmeta, atlases, font)
3. Mod Development (Modern Fabric & NeoForge, Mixin injection, registries, block entities, networking)
4. Redstone Engineering (RS tick vs Game tick, signal strength, comparators, quasi-connectivity, 0-tick, pulse generators, logic gates)
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

MINECRAFT_KNOWLEDGE_MODULES = [
    # ==================================================================
    # 1. DATA PACKS (MODERN 1.20.5+ / 1.21+)
    # ==================================================================
    {
        "doc_id": "mc_datapack_components_121",
        "title": "Minecraft 1.20.5+ / 1.21+ Item Component Architecture",
        "category": "datapack_syntax",
        "content": (
            "In Minecraft 1.20.5 and 1.21+, item NBT tags were completely removed and replaced by the structured "
            "'Item Components' system.\n\n"
            "Old syntax (deprecated/invalid):\n"
            "/give @p diamond_sword{Damage:10,Enchantments:[{id:\"minecraft:sharpness\",lvl:5s}]}\n\n"
            "New structured component syntax:\n"
            "/give @p diamond_sword[damage=10,enchantments={levels:{\"minecraft:sharpness\":5}}]\n\n"
            "Custom metadata must now be stored inside the 'minecraft:custom_data' component:\n"
            "/give @p stick[custom_data={spell_tier:3,element:\"fire\"}]\n\n"
            "Testing for components with execute if items:\n"
            "/execute if items entity @p weapon.mainhand diamond_sword[custom_data~{spell_tier:3}] run ...\n\n"
            "Architectural principle: Components are strongly typed, schema-validated at compile/load time, "
            "preventing silent NBT serialization corruptions."
        )
    },
    {
        "doc_id": "mc_datapack_macros_and_subroutines",
        "title": "Data Pack Macro Functions and Dynamic Subroutines",
        "category": "datapack_syntax",
        "content": (
            "Minecraft 1.20.2+ introduced Function Macros, which allow mcfunction files to accept dynamic arguments "
            "from storage or command parameters.\n\n"
            "Macro declaration: Any line starting with '$' treats '$(variable_name)' as a substituted token.\n"
            "Example 'example:spawn_custom_mob.mcfunction':\n"
            "$summon $(mob_type) ~ ~ ~ {CustomName:'{\"text\":\"$(name)\",\"color\":\"$(color)\"}'}\n\n"
            "Invocation via function command with compound source:\n"
            "/function example:spawn_custom_mob with storage example:params mob_config\n"
            "Or directly from inline macro compound arguments:\n"
            "/function example:spawn_custom_mob {mob_type:\"minecraft:zombie\",name:\"Lord Bones\",color:\"red\"}\n\n"
            "Execution model: Macro lines are parsed and compiled on-the-fly when executed, incurring a slight runtime "
            "overhead compared to pre-compiled static commands. Avoid macro execution inside per-tick entity loops (20 TPS)."
        )
    },
    {
        "doc_id": "mc_datapack_predicates_and_loot",
        "title": "Predicates, Damage Types, and Loot Tables",
        "category": "datapack_syntax",
        "content": (
            "Predicates (JSON files located in data/<namespace>/predicates/) define reusable condition checks executed on the server.\n"
            "Common conditions include 'minecraft:entity_properties', 'minecraft:location_check', 'minecraft:random_chance', "
            "and 'minecraft:time_check'.\n\n"
            "Damage types (1.20+ data/<namespace>/damage_type/) decouple damage logic from hardcoded mechanics. "
            "Components include 'exhaustion', 'scaling' (when_caused_by_living_non_player), and 'effects' (hurt, thorns, drown).\n\n"
            "Loot tables (data/<namespace>/loot_tables/) control item drops, chest generation, and fishing loot. "
            "In 1.20.5+, loot functions like 'minecraft:set_components' replace the legacy 'set_nbt' function."
        )
    },

    # ==================================================================
    # 2. RESOURCE PACKS
    # ==================================================================
    {
        "doc_id": "mc_resourcepack_blockstates_models",
        "title": "Resource Pack Blockstate Mapping and Model Inheritance",
        "category": "resourcepack_syntax",
        "content": (
            "A resource pack maps in-game block states to 3D rendering geometries through two JSON layers:\n"
            "1. 'assets/<namespace>/blockstates/<block>.json': Maps block state properties (e.g., 'facing=north,lit=true') "
            "to specific model files, UV rotations (x: 0/90/180/270, y: 0/90/180/270), and multipart conditions.\n"
            "2. 'assets/<namespace>/models/block/<model>.json': Defines geometric elements [from, to], faces (texture references, "
            "cullface, uv), and display transformations (thirdperson_righthand, gui, ground).\n\n"
            "Cullface optimization rule: Setting 'cullface': '<direction>' ensures that if an adjacent opaque block is present, "
            "the corresponding face is culled by the GPU, preserving draw call efficiency.\n\n"
            "pack.mcmeta format:\n"
            "{\"pack\": {\"pack_format\": 34, \"description\": \"Technical Resource Pack for 1.21\"}}"
        )
    },
    {
        "doc_id": "mc_resourcepack_atlases_and_fonts",
        "title": "Texture Atlases, Core Shaders, and Custom TTF/Bitmap Fonts",
        "category": "resourcepack_syntax",
        "content": (
            "Minecraft 1.19.3+ introduced Texture Atlases ('assets/<namespace>/atlases/blocks.json').\n"
            "Atlases stitch multiple loose PNG files into unified GPU texture sheets using sources:\n"
            "- 'directory': recursively scans folders.\n"
            "- 'single': stitches an explicit texture.\n"
            "- 'filter': excludes files.\n\n"
            "Custom fonts ('assets/<namespace>/font/default.json') support bitmap providers and 'ttf' (TrueType Font) providers. "
            "TTF providers allow scalable vector fonts with configurable shift, size, and oversample parameters."
        )
    },

    # ==================================================================
    # 3. MOD DEVELOPMENT (FABRIC & NEOFORGE)
    # ==================================================================
    {
        "doc_id": "mc_mod_fabric_mixins",
        "title": "Modern Fabric Modding: Lifecycle, Registries, and Mixin Injection",
        "category": "mod_development",
        "content": (
            "Fabric mod development relies on lightweight bytecode transformation via SpongePowered Mixin.\n\n"
            "1. Entrypoint: Implements 'net.fabricmc.api.ModInitializer' with 'onInitialize()'.\n"
            "2. Registries: Content is registered into 'net.minecraft.core.registries.BuiltInRegistries' via:\n"
            "   Registry.register(BuiltInRegistries.ITEM, Identifier.of(\"mymod\", \"custom_item\"), CUSTOM_ITEM);\n\n"
            "3. Mixin injection semantics:\n"
            "@Mixin(ServerPlayer.class)\n"
            "public class ServerPlayerMixin {\n"
            "    @Inject(method = \"tick\", at = @At(\"HEAD\"), cancellable = true)\n"
            "    private void onPlayerTick(CallbackInfo ci) {\n"
            "        // Executed at method entry\n"
            "        if (someCondition) { ci.cancel(); }\n"
            "    }\n"
            "}\n\n"
            "Pitfall: In 1.20.5+, 'FabricItemSettings' was removed in favor of standard 'Item.Settings'."
        )
    },
    {
        "doc_id": "mc_mod_neoforge_architecture",
        "title": "NeoForge Architecture: Event Bus, Capabilities, and DeferredRegister",
        "category": "mod_development",
        "content": (
            "NeoForge utilizes an event-driven decoupled architecture:\n"
            "1. Mod Lifecycle: Annotated with '@Mod(\"mod_id\")'. Listens to the mod event bus (IEventBus) for setup events.\n"
            "2. DeferredRegister: Thread-safe deferred registration:\n"
            "   public static final DeferredRegister.Items ITEMS = DeferredRegister.createItems(\"mod_id\");\n"
            "   public static final DeferredItem<Item> RUBY = ITEMS.registerSimpleItem(\"ruby\", new Item.Properties());\n\n"
            "3. NeoForge Capabilities (1.20.2+): Replaced old Forge capabilities with unified BlockCapability, "
            "EntityCapability, and ItemCapability. Providers are registered via 'RegisterCapabilitiesEvent', "
            "allowing items and blocks to query energy, item storage, and fluid transfer without direct coupling."
        )
    },

    # ==================================================================
    # 4. REDSTONE ENGINEERING (RS CIRCUITS & LOGIC)
    # ==================================================================
    {
        "doc_id": "mc_redstone_timing_and_signals",
        "title": "Redstone Timing: Game Ticks, Redstone Ticks, and Signal Attenuation",
        "category": "redstone_circuit",
        "content": (
            "Redstone signal fundamentals:\n"
            "1. Signal Strength: Ranges from 0 (off) to 15 (max). Attenuates by 1 per block of redstone dust traversed. "
            "Repeaters restore signal strength to 15, introducing a 1-to-4 redstone tick delay.\n"
            "2. Timing hierarchy:\n"
            "   - 1 Game Tick (gt) = 0.05 seconds (20 gt/second).\n"
            "   - 1 Redstone Tick (rt) = 2 Game Ticks = 0.1 seconds (10 rt/second).\n"
            "   - Repeater delays: 1rt (2gt), 2rt (4gt), 3rt (6gt), 4rt (8gt).\n"
            "   - Observer: Emits a 1-redstone-tick (2 game ticks) pulse upon detecting a block state or block update.\n"
            "   - Redstone Torch: Inverts signal with a 1rt (2gt) delay. Rapidly cycling (< 8 toggles in 60 ticks) triggers burnout."
        )
    },
    {
        "doc_id": "mc_redstone_comparators_math",
        "title": "Comparator Operations: Subtraction, Comparison, and Container Physics",
        "category": "redstone_circuit",
        "content": (
            "The Redstone Comparator operates in two distinct modes:\n"
            "1. Comparison Mode (front torch unlit): If rear input >= maximum of side inputs, output = rear input; else output = 0.\n"
            "2. Subtraction Mode (front torch lit): Output = max(0, rear input - max(side input A, side input B)).\n\n"
            "Container measurement formula:\n"
            "Signal Strength = floor(1 + ((sum over slots of (item_count / max_stack_size)) / total_slots) * 14)\n"
            "If container is empty, output is exactly 0. Even 1 item in a large container yields a minimum output of 1.\n\n"
            "Application: Pulse length extenders, analog subtraction arithmetic, and exact item-sorter filters (e.g. 41 items in slot 1)."
        )
    },
    {
        "doc_id": "mc_redstone_quasi_connectivity_and_zero_tick",
        "title": "Quasi-Connectivity (QC), BUDs, and 0-Tick Piston Spitting Mechanics",
        "category": "redstone_circuit",
        "content": (
            "Java Edition specific physical mechanics:\n"
            "1. Quasi-Connectivity (QC): Pistons, droppers, and dispensers check for redstone power as if they were a 2-block-tall "
            "door (checking the space directly above them). If that upper space is powered by redstone dust or a block, the piston "
            "becomes powered, but does NOT actuate until it receives a direct Block Update (BUD: Block Update Detector).\n\n"
            "2. Sticky Piston Block Spitting (0-Tick and 1-Tick Pulses):\n"
            "When a sticky piston receives a 1-redstone-tick pulse (or a microtick/0-tick pulse), it extends and retracts so rapidly "
            "that it leaves its attached block behind (spits the block) instead of pulling it back. A subsequent 1-tick pulse retrieves it.\n"
            "This property forms the foundation of T-Flip-Flops, fast piston doors (Jeb doors, 3x3 doors), and flying machines."
        )
    },
    {
        "doc_id": "mc_redstone_logic_gates_and_computing",
        "title": "Redstone Logic Gates, Binary Adders, and Directionality",
        "category": "redstone_circuit",
        "content": (
            "Digital logic implementations in Minecraft:\n"
            "1. NOT Gate: Redstone torch on the side of a solid block powered by input wire.\n"
            "2. AND Gate: Two torches on separate input blocks powering a common redstone wire that inverts a third torch.\n"
            "3. XOR Gate: Constructed using dual comparator subtraction loops or torch multiplexers.\n"
            "4. Full Adder: Cascaded XOR gates with carry generation (A AND B) or (Carry AND (A XOR B)).\n\n"
            "Directionality & Locationality: Block updates follow strict coordinate processing order (hash set iteration). "
            "Circuits that rely on simultaneous tile tick updates can exhibit orientation-dependent behavior across chunk boundaries."
        )
    }
]

def main():
    output_dir = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/minecraft_corpus"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "minecraft_technical_knowledge_corpus.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(MINECRAFT_KNOWLEDGE_MODULES, f, indent=2, ensure_ascii=False)
        
    print(f"[CorpusBuilder] Successfully built and saved {len(MINECRAFT_KNOWLEDGE_MODULES)} modules to {json_path}")

if __name__ == "__main__":
    main()
