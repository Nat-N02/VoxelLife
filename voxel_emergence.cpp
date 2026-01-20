// voxel_emergence_opt.cpp
// Optimized single-thread CPU baseline with:
// - curr->next only
// - precomputed neighbors
// - prehashed voxel IDs for cheap deterministic RNG
// - persistent scratch arrays
//
// Changes included:
// 1) Damage-dependent global leak (dead matter hemorrhages energy)
// 2) Repair scarcity: thresholded, gated, efficiency falloff, superlinear cost
// 3) Expanded diagnostics: variance + percentiles + repair/transport/junction metrics

#include <numeric>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <sstream>

// ============================================================
// Utility
// ============================================================
static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}
static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// ============================================================
// Directions (6-neighborhood)
// ============================================================
enum Dir : int { XP=0, XN=1, YP=2, YN=3, ZP=4, ZN=5 };

static inline Dir opposite(Dir d) {
    switch (d) {
        case XP: return XN; case XN: return XP;
        case YP: return YN; case YN: return YP;
        case ZP: return ZN; case ZN: return ZP;
        default: return XP;
    }
}

// ============================================================
// Deterministic RNG (SplitMix64)
// ============================================================
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // [-1,1)
    float next_f11() {
        uint32_t x = static_cast<uint32_t>(next_u64() >> 40);
        float f01 = (x + 0.5f) * (1.0f / 16777216.0f);
        return 2.0f * f01 - 1.0f;
    }
};

static inline uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

struct LagBuffer {
    std::vector<double> R;   // mean repair activity per tick
    std::vector<double> F;   // mean flux per tick
    int maxlag;
};

static constexpr uint64_t MIX_TICK  = 0x9E3779B97F4A7C15ull;
static constexpr uint64_t MIX_PAIR  = 0xD1B54A32D192ED03ull;
static constexpr uint64_t MIX_DIR   = 0x94D049BB133111EBull;

static inline float hash_to_f11(uint64_t x) {
    uint64_t h = hash_u64(x);
    uint32_t top = (uint32_t)(h >> 40);
    float f01 = (top + 0.5f) * (1.0f / 16777216.0f);
    return 2.0f * f01 - 1.0f;
}

// ============================================================
// Params
// ============================================================
struct Params {
    int nx=128, ny=128, nz=4;

    float E_global_leak = 0.01f;
    float E_route_loss  = 0.1f;

    float D_ref = 10.0f;
    float D_activity_gain = 0.025f;
    float D_slow_anneal = 0.00015f; // 0 no effect, 0.0005 melts

    float D_to_conduct_drop = 0.3704f; // 0 seems to do nothing? 100 makes some odd crystal thing
    float D_to_noise = 0.03f;

    float W_floor = 1e-4f; // little effect until large scales
    float W_decay = 0.0f; // 0.001 seems to expand? 0 inhibits development

    float repair_strength = 0.05f; // 0.025 inhibits, 0 shrink away to nothing, 0.1 rapid expand
    float repair_energy_cost = 0.05f; // 0 lowers damage and increases size, 0.1 adapts
    float repair_noise = 0.2f; // generally low effect

    float porous_conduct_floor = 0.01f;
    float porous_noise_boost = 0.5f;

    float repair_boost_decay = 0.9995f;

    float repair_surface_k = 0.3f;       // strength of surface penalty
    float repair_surface_power = 1.2f;   // 1 = linear, 2 = harsh

    // --- blob splitting (damage wall) ---
    bool enable_cut = false;
    int cut_start_tick = 16000;
    int cut_end_tick   = 17000;          // duration ~5000 ticks
    int cut_axis = 1;                    // 0=X, 1=Y, 2=Z
    float cut_frac = 0.5f;              // where the plane is (0..1)
    float cut_thickness = 10.0f;          // in voxels
    float cut_damage = 0.8f;             // fraction of D_ref

    float sent_tail_rise  = 0.20f;  
    float sent_tail_decay = 0.995f; 
    float repair_tail_frac = 0.75f;       // IMPORTANT: 0.6-0.7 blob range
    int sent_tail_radius = 8;

    float source_inject = 0.05f;

    float leak_k = 8.0f; 
    float repair_cost_activity_k = 1.0f;  
    float repair_cost_super_k = 20.0f;   

    // --- Precursor -> R_boost ---
    float P0 = 1.0f;            // baseline precursor level
    float P_alpha = 0.005f;    // relaxation rate toward P0

    float P_damage_gain = 0.0f;   // 0 = old behavior, 2–5 is reasonable

    float k_PR = 0.002f;         // conversion rate (k)
    float R_sat = 1.0f;         // saturation scale (RSAT): larger = higher local ceiling

    float act_thresh = 0.15f;   // needs to be routing ~30% of stored energy
    float act_k      = 5.0f;  // sharp but not binary

    float R_spend_k = 0.01f;     // spend rate per unit repair amount

    // diagnostics
    int print_every = 1000;
    bool load_from_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            std::cerr << "Could not open params file: " << path << "\n";
            return false;
        }

        std::unordered_map<std::string, std::string> kv;
        std::string line;

        while (std::getline(f, line)) {
            // strip comments
            auto hash = line.find('#');
            if (hash != std::string::npos)
                line = line.substr(0, hash);

            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;

            auto trim = [](std::string s) {
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of(" \t\r\n");
                if (a == std::string::npos) return std::string();
                return s.substr(a, b - a + 1);
            };

            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));

            if (!key.empty())
                kv[key] = val;
        }

        auto getf = [&](const char* k, float& x) {
            if (kv.count(k)) x = std::stof(kv[k]);
        };
        auto geti = [&](const char* k, int& x) {
            if (kv.count(k)) x = std::stoi(kv[k]);
        };
        auto getb = [&](const char* k, bool& x) {
            if (kv.count(k)) {
                std::string v = kv[k];
                x = (v == "1" || v == "true" || v == "True");
            }
        };

        // Grid
        geti("nx", nx);
        geti("ny", ny);
        geti("nz", nz);

        // Energy
        getf("E_global_leak", E_global_leak);
        getf("E_route_loss", E_route_loss);
        getf("source_inject", source_inject);

        // Damage
        getf("D_ref", D_ref);
        getf("D_activity_gain", D_activity_gain);
        getf("D_slow_anneal", D_slow_anneal);
        getf("D_to_conduct_drop", D_to_conduct_drop);
        getf("D_to_noise", D_to_noise);

        // Weights
        getf("W_floor", W_floor);
        getf("W_decay", W_decay);
        getf("porous_conduct_floor", porous_conduct_floor);
        getf("porous_noise_boost", porous_noise_boost);

        // Repair
        getf("repair_strength", repair_strength);
        getf("repair_energy_cost", repair_energy_cost);
        getf("repair_noise", repair_noise);
        getf("repair_boost_decay", repair_boost_decay);
        getf("repair_surface_k", repair_surface_k);
        getf("repair_surface_power", repair_surface_power);
        getf("repair_tail_frac", repair_tail_frac);
        geti("sent_tail_radius", sent_tail_radius);
        getf("R_spend_k", R_spend_k);

        // Precursor
        getf("P0", P0);
        getf("P_alpha", P_alpha);
        getf("P_damage_gain", P_damage_gain);
        getf("k_PR", k_PR);
        getf("R_sat", R_sat);
        getf("act_thresh", act_thresh);
        getf("act_k", act_k);

        // Leakage / cost
        getf("leak_k", leak_k);
        getf("repair_cost_activity_k", repair_cost_activity_k);
        getf("repair_cost_super_k", repair_cost_super_k);

        // Diagnostics
        geti("print_every", print_every);

        return true;
    }

    bool set_param(const std::string& key, const std::string& val) {
        auto to_f = [&](float& x) { x = std::stof(val); };
        auto to_i = [&](int& x) { x = std::stoi(val); };
        auto to_b = [&](bool& x) {
            x = (val == "1" || val == "true" || val == "True");
        };

        // Grid
        if (key == "nx") { to_i(nx); return true; }
        if (key == "ny") { to_i(ny); return true; }
        if (key == "nz") { to_i(nz); return true; }

        // Energy
        if (key == "E_global_leak") { to_f(E_global_leak); return true; }
        if (key == "E_route_loss") { to_f(E_route_loss); return true; }
        if (key == "source_inject") { to_f(source_inject); return true; }

        // Damage
        if (key == "D_ref") { to_f(D_ref); return true; }
        if (key == "D_activity_gain") { to_f(D_activity_gain); return true; }
        if (key == "D_slow_anneal") { to_f(D_slow_anneal); return true; }
        if (key == "D_to_conduct_drop") { to_f(D_to_conduct_drop); return true; }
        if (key == "D_to_noise") { to_f(D_to_noise); return true; }

        // Weights
        if (key == "W_floor") { to_f(W_floor); return true; }
        if (key == "W_decay") { to_f(W_decay); return true; }
        if (key == "porous_conduct_floor") { to_f(porous_conduct_floor); return true; }
        if (key == "porous_noise_boost") { to_f(porous_noise_boost); return true; }

        // Repair
        if (key == "repair_strength") { to_f(repair_strength); return true; }
        if (key == "repair_energy_cost") { to_f(repair_energy_cost); return true; }
        if (key == "repair_noise") { to_f(repair_noise); return true; }
        if (key == "repair_boost_decay") { to_f(repair_boost_decay); return true; }
        if (key == "repair_surface_k") { to_f(repair_surface_k); return true; }
        if (key == "repair_surface_power") { to_f(repair_surface_power); return true; }
        if (key == "repair_tail_frac") { to_f(repair_tail_frac); return true; }
        if (key == "sent_tail_radius") { to_i(sent_tail_radius); return true; }
        if (key == "R_spend_k") { to_f(R_spend_k); return true; }

        // Precursor
        if (key == "P0") { to_f(P0); return true; }
        if (key == "P_alpha") { to_f(P_alpha); return true; }
        if (key == "P_damage_gain") { to_f(P_damage_gain); return true; }
        if (key == "k_PR") { to_f(k_PR); return true; }
        if (key == "R_sat") { to_f(R_sat); return true; }
        if (key == "act_thresh") { to_f(act_thresh); return true; }
        if (key == "act_k") { to_f(act_k); return true; }

        // Leakage / cost
        if (key == "leak_k") { to_f(leak_k); return true; }
        if (key == "repair_cost_activity_k") { to_f(repair_cost_activity_k); return true; }
        if (key == "repair_cost_super_k") { to_f(repair_cost_super_k); return true; }

        // Diagnostics
        if (key == "print_every") { to_i(print_every); return true; }

        return false;
    }

};

// ============================================================
// Field (SoA)
// ============================================================
struct Field {
    std::vector<float> E;
    std::vector<float> D;
    std::vector<float> W; // packed [i*6 + d]
    std::vector<float> P; // precursor

    void resize(size_t n) {
        E.assign(n, 0.0f);
        D.assign(n, 0.0f);
        W.assign(n*6, 0.0f);
        P.assign(n, 0.0f);
    }
};

// ============================================================
// World
// ============================================================
struct World {
    Params p;
    size_t nvox;

    Field curr, next;

    // Persistent scratch
    std::vector<float> sent;          // total sent per voxel (activity)
    std::vector<float> sent_dir;      // per-direction sent (size nvox*6)
    std::vector<float> repair_delta;  // damage repair applied to target
    std::vector<float> repair_cost;   // energy cost paid by repairer (dissipated)
    std::vector<float> repair_elig;
    std::vector<float> R_boost;
    std::vector<float> E_residual;
    std::vector<float> D_prev_tick;
    std::vector<int> top1_age, top5_age;
    std::vector<float> sent_tail_local;  // size nvox
    std::vector<float> tmpX, tmpY;        // scratch for separable max-filter
    std::vector<uint8_t> topR_prev;
    bool have_topR_prev = false;

    std::string metrics_path = "regime_metrics.csv";
    bool metrics_written = false;

    std::vector<uint8_t> top1_prev, top5_prev, top10_prev;
    bool have_top_prev = false;

    // Precomputed neighbors (size nvox*6), -1 = out of bounds
    std::vector<int> nbr;

    // Precomputed per-voxel hash base (size nvox)
    std::vector<uint64_t> vh;

    // --- RFC ---
    LagBuffer rfc_buf { {}, {}, 8 };   // tau_max = 8 is plenty

    // --- EMU ---
    std::vector<uint8_t> fossil_C;     // frozen crystal imprint
    bool fossil_frozen = false;

    uint64_t seed;
    uint64_t tick = 0;

    // Per-tick diagnostic counters (reset each step)
    int repair_events_tick = 0;
    int repair_eligible_tick = 0;
    double routed_sum_tick = 0.0;
    int routed_active_voxels_tick = 0;
    float sent_q95_cached = 0.0f;
    float sent_q99_cached = 0.0f;

    // --- Temporal diagnostics ---
    std::vector<float> D_prev;
    double Dm_prev = 0.0;
    bool have_prev_snapshot = false;


    explicit World(const Params& params, uint64_t seed_)
        : p(params), seed(seed_) {

        nvox = static_cast<size_t>(p.nx) * p.ny * p.nz;
        curr.resize(nvox);
        next.resize(nvox);

        sent.assign(nvox, 0.0f);
        sent_dir.assign(nvox*6, 0.0f);
        repair_delta.assign(nvox, 0.0f);
        repair_cost.assign(nvox, 0.0f);
        repair_elig.assign(nvox, 0.0f);
        R_boost.assign(nvox, 0.0f);
        E_residual.assign(nvox, 0.0f);
        top1_prev.resize(nvox, 0);
        top5_prev.resize(nvox, 0);
        top10_prev.resize(nvox, 0);
        top1_age.assign(nvox, 0);
        top5_age.assign(nvox, 0);
        sent_tail_local.assign(nvox, 1e-6f);    
        tmpX.assign(nvox, 0.0f);
        tmpY.assign(nvox, 0.0f);
        fossil_C.assign(nvox, 0);
        topR_prev.assign(nvox, 0);
        have_topR_prev = false;

        D_prev_tick.assign(nvox, 0.0f);

        nbr.assign(nvox*6, -1);
        vh.assign(nvox, 0);
        D_prev.assign(nvox, 0.0f);

        build_neighbors_and_hashes();
        init_state();
    }

    inline int idx(int x, int y, int z) const {
        return (z * p.ny + y) * p.nx + x;
    }

    inline bool in_cut_region(int x, int y, int z) const {
        if (!p.enable_cut) return false;
        if (tick < (uint64_t)p.cut_start_tick || tick >= (uint64_t)p.cut_end_tick)
            return false;

        float coord = 0.0f;
        float center = 0.0f;

        if (p.cut_axis == 0) { coord = float(x); center = p.cut_frac * (p.nx - 1); }
        if (p.cut_axis == 1) { coord = float(y); center = p.cut_frac * (p.ny - 1); }
        if (p.cut_axis == 2) { coord = float(z); center = p.cut_frac * (p.nz - 1); }

        return std::abs(coord - center) <= p.cut_thickness;
    }


    // --------------------------------------------------------
    // Precompute neighbors & voxel hashes
    // --------------------------------------------------------
    void build_neighbors_and_hashes() {
        for (int z=0; z<p.nz; z++) {
            for (int y=0; y<p.ny; y++) {
                for (int x=0; x<p.nx; x++) {
                    int i = idx(x,y,z);

                    nbr[i*6 + XP] = (x+1 < p.nx) ? idx(x+1,y,z) : -1;
                    nbr[i*6 + XN] = (x-1 >= 0)   ? idx(x-1,y,z) : -1;
                    nbr[i*6 + YP] = (y+1 < p.ny) ? idx(x,y+1,z) : -1;
                    nbr[i*6 + YN] = (y-1 >= 0)   ? idx(x,y-1,z) : -1;
                    nbr[i*6 + ZP] = (z+1 < p.nz) ? idx(x,y,z+1) : -1;
                    nbr[i*6 + ZN] = (z-1 >= 0)   ? idx(x,y,z-1) : -1;
                }
            }
        }

        for (size_t i=0; i<nvox; i++) {
            vh[i] = hash_u64(seed ^ (uint64_t)i);
        }
    }

    void append_metrics_csv(
        double BSI,
        double RFC,
        double EMU,
        double SPI,
        float D_q50,
        float D_q95,
        float sent_q95
    ) {
        bool exists = false;
        {
            std::ifstream f(metrics_path);
            exists = f.good();
        }

        std::ofstream out(metrics_path, std::ios::app);

        // Header (only once)
        if (!exists) {
            out
                << "seed,nx,ny,nz,"
                << "sent_tail_radius,repair_tail_frac,W_decay,"
                << "BSI,RFC,EMU,SPI,"
                << "D_q50,D_q95,sent_q95\n";
        }

        out
            << seed << ","
            << p.nx << "," << p.ny << "," << p.nz << ","
            << p.sent_tail_radius << ","
            << p.repair_tail_frac << ","
            << p.W_decay << ","
            << BSI << ","
            << RFC << ","
            << EMU << ","
            << SPI << ","
            << D_q50 << ","
            << D_q95 << ","
            << sent_q95
            << "\n";
    }


    // --------------------------------------------------------
    // Dominant axis computed from curr.W only
    // --------------------------------------------------------
    struct AxisInfo { Dir fwd, back, axis; };

    // ============================================================
    // Checkpoint I/O
    // ============================================================
    static constexpr uint32_t CKPT_MAGIC = 0x564F5843u; // 'VOXC'
    static constexpr uint32_t CKPT_VER   = 1;

    template <typename T>
    static void write_vec(std::ostream& os, const std::vector<T>& v) {
        uint64_t n = (uint64_t)v.size();
        os.write((const char*)&n, sizeof(n));
        if (n) os.write((const char*)v.data(), sizeof(T) * n);
    }

    template <typename T>
    static void read_vec(std::istream& is, std::vector<T>& v) {
        uint64_t n = 0;
        is.read((char*)&n, sizeof(n));
        v.resize((size_t)n);
        if (n) is.read((char*)v.data(), sizeof(T) * n);
    }

    bool save_checkpoint(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) return false;

        // header
        os.write((const char*)&CKPT_MAGIC, sizeof(CKPT_MAGIC));
        os.write((const char*)&CKPT_VER,   sizeof(CKPT_VER));

        // dims / tick
        os.write((const char*)&p.nx, sizeof(p.nx));
        os.write((const char*)&p.ny, sizeof(p.ny));
        os.write((const char*)&p.nz, sizeof(p.nz));
        os.write((const char*)&tick, sizeof(tick));
        os.write((const char*)&seed, sizeof(seed));

        // --- state that defines dynamics ---
        write_vec(os, curr.E);
        write_vec(os, curr.D);
        write_vec(os, curr.W);
        write_vec(os, curr.P);

        write_vec(os, repair_elig);
        write_vec(os, R_boost);

        // If you have E_residual as a persistent member, include it:
        // write_vec(os, E_residual);

        // --- optional: temporal diagnostics continuity ---
        uint8_t hp = have_prev_snapshot ? 1 : 0;
        os.write((const char*)&hp, sizeof(hp));
        os.write((const char*)&Dm_prev, sizeof(Dm_prev));
        write_vec(os, D_prev);

        // Add any other “history” vectors you created (topK prev/ages) here.

        return (bool)os;
    }

    bool load_checkpoint(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) return false;

        uint32_t magic=0, ver=0;
        is.read((char*)&magic, sizeof(magic));
        is.read((char*)&ver,   sizeof(ver));
        if (magic != CKPT_MAGIC || ver != CKPT_VER) return false;

        int nx=0, ny=0, nz=0;
        is.read((char*)&nx, sizeof(nx));
        is.read((char*)&ny, sizeof(ny));
        is.read((char*)&nz, sizeof(nz));

        // IMPORTANT: if dims differ, you either reject or rebuild everything.
        if (nx != p.nx || ny != p.ny || nz != p.nz) {
            std::cerr << "Checkpoint dims mismatch: "
                    << nx << "," << ny << "," << nz
                    << " vs params " << p.nx << "," << p.ny << "," << p.nz << "\n";
            return false;
        }

        is.read((char*)&tick, sizeof(tick));
        is.read((char*)&seed, sizeof(seed));

        // Load state vectors
        read_vec(is, curr.E);
        read_vec(is, curr.D);
        read_vec(is, curr.W);
        read_vec(is, curr.P);

        read_vec(is, repair_elig);
        read_vec(is, R_boost);

        // If you saved E_residual, read it too:
        // read_vec(is, E_residual);

        // Optional temporal continuity
        uint8_t hp=0;
        is.read((char*)&hp, sizeof(hp));
        have_prev_snapshot = (hp != 0);
        is.read((char*)&Dm_prev, sizeof(Dm_prev));
        read_vec(is, D_prev);

        // Now re-derive nvox and ensure scratch is sized.
        nvox = (size_t)p.nx * p.ny * p.nz;

        auto must = [&](auto& v, const char* name) {
            if (v.size() != nvox) {
                std::cerr << "Checkpoint vector size mismatch: " << name
                        << " size=" << v.size() << " nvox=" << nvox << "\n";
                std::abort();
            }
        };

        must(curr.E, "curr.E");
        must(curr.D, "curr.D");
        auto must_nvox = [&](const std::vector<float>& v, const char* name) {
            if (v.size() != nvox) {
                std::cerr << "Checkpoint vector size mismatch: " << name
                        << " size=" << v.size() << " nvox=" << nvox << "\n";
                std::abort();
            }
        };

        auto must_nvox6 = [&](const std::vector<float>& v, const char* name) {
            if (v.size() != nvox * 6) {
                std::cerr << "Checkpoint vector size mismatch: " << name
                        << " size=" << v.size() << " expected=" << (nvox * 6) << "\n";
                std::abort();
            }
        };
        must(repair_elig, "repair_elig");
        must(R_boost, "R_boost");
        if (have_prev_snapshot) must(D_prev, "D_prev");

        // Rebuild derived tables that depend on seed/dims if needed:
        // nbr/vh should already be correct for these dims,
        // but if seed changed you probably want to rebuild vh.
        // In your current setup seed is part of the checkpoint, so:
        build_neighbors_and_hashes();

        // Ensure next and scratch arrays exist
        next.resize(nvox);
        sent.assign(nvox, 0.0f);
        sent_dir.assign(nvox*6, 0.0f);
        repair_delta.assign(nvox, 0.0f);
        repair_cost.assign(nvox, 0.0f);

        return (bool)is;
    }

    void dump_fields() const {
        char fname[256];
        std::snprintf(fname, sizeof(fname),
            "dump/dump_t%08llu.bin",
            (unsigned long long)tick
        );

        FILE* f = std::fopen(fname, "wb");
        if (!f) {
            std::perror("dump fopen failed");
            return;
        }

        int32_t nx = p.nx;
        int32_t ny = p.ny;
        int32_t nz = p.nz;

        std::fwrite(&nx, sizeof(int32_t), 1, f);
        std::fwrite(&ny, sizeof(int32_t), 1, f);
        std::fwrite(&nz, sizeof(int32_t), 1, f);
        std::fwrite(&tick, sizeof(uint64_t), 1, f);

        std::fwrite(curr.E.data(), sizeof(float), nvox, f);
        std::fwrite(curr.D.data(), sizeof(float), nvox, f);
        std::fwrite(curr.P.data(), sizeof(float), nvox, f);
        std::fwrite(R_boost.data(), sizeof(float), nvox, f);

        std::fclose(f);
    }

    double compute_BSI(const std::vector<float>& D,
                       float D_q50,
                       const std::vector<int>& nbr,
                       size_t nvox)
    {
        double grad_sum = 0.0;
        double grad_boundary_sum = 0.0;
        int grad_count = 0;
        int boundary_count = 0;

        for (size_t i = 0; i < nvox; i++) {
            float Di = D[i];
            bool Mi = (Di < D_q50);

            float gx = 0.0f;
            int ncnt = 0;

            for (int d = 0; d < 6; d++) {
                int j = nbr[i*6 + d];
                if (j < 0) continue;
                gx += std::abs(D[j] - Di);
                ncnt++;
            }
            if (ncnt == 0) continue;

            float grad = gx / float(ncnt);
            grad_sum += grad;
            grad_count++;

            bool is_boundary = false;
            for (int d = 0; d < 6; d++) {
                int j = nbr[i*6 + d];
                if (j < 0) continue;
                if ((D[j] < D_q50) != Mi) {
                    is_boundary = true;
                    break;
                }
            }

            if (is_boundary) {
                grad_boundary_sum += grad;
                boundary_count++;
            }
        }

        if (grad_count == 0 || boundary_count == 0) return 0.0;
        return (grad_boundary_sum / boundary_count) / (grad_sum / grad_count);
    }

    double compute_RFC(const LagBuffer& buf) {
        int T = buf.R.size();
        if (T < 2) return 0.0;

        double best = 0.0;
        for (int tau = 0; tau <= buf.maxlag; tau++) {
            if (T <= tau + 1) continue;

            double mr = 0.0, mf = 0.0;
            int n = T - tau;
            for (int i = tau; i < T; i++) {
                mr += buf.R[i];
                mf += buf.F[i - tau];
            }
            mr /= n; mf /= n;

            double num = 0.0, dr = 0.0, df = 0.0;
            for (int i = tau; i < T; i++) {
                double a = buf.R[i] - mr;
                double b = buf.F[i - tau] - mf;
                num += a * b;
                dr += a * a;
                df += b * b;
            }
            if (dr > 0 && df > 0) {
                best = std::max(best, num / std::sqrt(dr * df));
            }
        }
        return best;
    }

    double compute_EMU(const std::vector<float>& D,
                       const std::vector<uint8_t>& fossil_C,
                       const std::vector<uint8_t>& repaired_flag,
                       float D_q50,
                       size_t nvox)
    {
        int Bcnt = 0;
        int BCcnt = 0;

        for (size_t i = 0; i < nvox; i++) {
            bool B = (D[i] < D_q50) && repaired_flag[i];
            if (!B) continue;
            Bcnt++;
            if (fossil_C[i]) BCcnt++;
        }
        if (Bcnt == 0) return 0.0;
        return double(BCcnt) / double(Bcnt);
    }

    inline AxisInfo dominant_axis(size_t i) const {
        const float* w = &curr.W[i*6];
        float wxp = w[XP], wxn = w[XN];
        float wyp = w[YP], wyn = w[YN];
        float wzp = w[ZP], wzn = w[ZN];

        float xbest = (wxp > wxn) ? wxp : wxn;
        float ybest = (wyp > wyn) ? wyp : wyn;
        float zbest = (wzp > wzn) ? wzp : wzn;

        if (xbest >= ybest && xbest >= zbest) {
            Dir f = (wxp >= wxn) ? XP : XN;
            return {f, opposite(f), XP};
        } else if (ybest >= xbest && ybest >= zbest) {
            Dir f = (wyp >= wyn) ? YP : YN;
            return {f, opposite(f), YP};
        } else {
            Dir f = (wzp >= wzn) ? ZP : ZN;
            return {f, opposite(f), ZP};
        }
    }

    inline float conductivity(float D) const {
        float x = clampf(D / p.D_ref, 0.0f, 1.0f);
        constexpr float x_crit = 0.6f;
        float sharpness = p.D_to_conduct_drop;
        float cliff = 1.0f / (1.0f + std::exp(sharpness * (x - x_crit)));
        float floor = p.porous_conduct_floor;
        return floor + (1.0f - floor) * cliff;
    }

    inline bool is_source_voxel_idx(size_t i) const {
        int x = int(i % p.nx);
        int y = int((i / p.nx) % p.ny);
        int z = int(i / (p.nx * p.ny));
        return is_source_voxel(x, y, z);
    }

    inline float local_D_gradient(size_t i) const {
        float Di = curr.D[i];
        float g = 0.0f;
        for (int d = 0; d < 6; d++) {
            int j = nbr[i*6 + d];
            if (j < 0) continue;
            g += std::fabs(Di - curr.D[(size_t)j]);
        }
        return g;
    }

    // --------------------------------------------------------
    // Weight normalize (tiny floor + sum normalize)
    // --------------------------------------------------------
    inline void normalize6(float w[6]) const {
        float sum = 0.0f;
        for (int d=0; d<6; d++) {
            w[d] = (w[d] < p.W_floor) ? p.W_floor : w[d];
            sum += w[d];
        }
        float inv = (sum > 0.0f) ? (1.0f / sum) : (1.0f / 6.0f);
        for (int d=0; d<6; d++) w[d] *= inv;
    }

    // --------------------------------------------------------
    // Sources (example: z==0 plane)
    // --------------------------------------------------------
    inline bool is_source_voxel(int x,int y,int z) const {
        return (z == 0 || z == 2);
    }

    // --------------------------------------------------------
    // Init state
    // --------------------------------------------------------
    void init_state() {
        for (size_t i=0; i<nvox; i++) {
            SplitMix64 rng(vh[i]); // stable jitter

            float w[6];
            float sum = 0.0f;
            for (int d=0; d<6; d++) {
                float v = 1.0f + 0.01f * rng.next_f11();
                v = (v < p.W_floor) ? p.W_floor : v;
                w[d] = v;
                sum += v;
                curr.P[i] = p.P0;   // or 0.5f*p.P0
            }
            float inv = 1.0f / sum;
            for (int d=0; d<6; d++) curr.W[i*6 + d] = w[d] * inv;

            curr.E[i] = 0.0f;
            curr.D[i] = 0.0f;
        }
    }

    // --------------------------------------------------------
    // One tick
    // --------------------------------------------------------
    void step() {
        // reset per-tick diagnostics
        repair_events_tick = 0;
        repair_eligible_tick = 0;
        routed_sum_tick = 0.0;
        routed_active_voxels_tick = 0;
        D_prev_tick = curr.D;

        // clear scratch
        std::fill(sent.begin(), sent.end(), 0.0f);
        std::fill(sent_dir.begin(), sent_dir.end(), 0.0f);
        std::fill(repair_delta.begin(), repair_delta.end(), 0.0f);
        std::fill(repair_cost.begin(), repair_cost.end(), 0.0f);
        std::fill(E_residual.begin(), E_residual.end(), 0.0f);

        for (size_t i = 0; i < nvox; i++) {
            // copy state
            next.D[i] = curr.D[i];
            next.P[i] = curr.P[i];
            // IMPORTANT: next.E is a pure accumulator
            next.E[i] = 0.0f;
        }


        for (size_t i = 0; i < nvox; i++) {
            R_boost[i] *= p.repair_boost_decay;
            if (R_boost[i] < 1e-6f) R_boost[i] = 0.0f;
            // slow relaxation toward baseline
            float d = clampf(curr.D[i] / p.D_ref, 0.0f, 1.0f);

            // more damage → faster P replenishment
            float alpha_eff = p.P_alpha * (1.0f + p.P_damage_gain * d);

            next.P[i] += alpha_eff * (p.P0 - curr.P[i]);
        }

        route_energy_curr_to_next();
        update_sent_tail();
        apply_perpendicular_repair();
        evolve_weights_curr_to_next();

        // --------------------------------------------------------
        // TRANSIENT DAMAGE CUT
        // --------------------------------------------------------
        if (p.enable_cut &&
            tick >= (uint64_t)p.cut_start_tick &&
            tick <  (uint64_t)p.cut_end_tick)
        {
            for (int z = 0; z < p.nz; z++) {
                for (int y = 0; y < p.ny; y++) {
                    for (int x = 0; x < p.nx; x++) {
                        if (!in_cut_region(x,y,z)) continue;
                        size_t i = (size_t)idx(x,y,z);
                        next.E[i] += p.cut_damage * p.D_ref;
                    }
                }
            }
        }

        for (size_t i=0; i<nvox; i++) {
            float activity = sent[i] + repair_delta[i];
            float k_eff = p.D_slow_anneal * (1 - 0.001f * activity);
            next.D[i] *= (1-k_eff);
        }
        for (int z=0; z<p.nz; z++) {
            for (int y=0; y<p.ny; y++) {
                for (int x=0; x<p.nx; x++) {
                    if (!is_source_voxel(x,y,z)) continue;
                    size_t i = (size_t)idx(x,y,z);
                    next.E[i] += p.source_inject;
                }
            }
        }

        for (size_t i=0; i<nvox; i++) {
            next.E[i] = clampf(next.E[i], 0.0f, 10000);
            next.D[i] = clampf(next.D[i], 0.0f, 10000);
        }

        std::swap(curr.E, next.E);
        std::swap(curr.D, next.D);
        std::swap(curr.W, next.W);
        std::swap(curr.P, next.P);

        tick++;
    }

    // --------------------------------------------------------
    // ROUTING: curr->next only (scatter), record sent + sent_dir
    // --------------------------------------------------------
    void route_energy_curr_to_next() {
        const float loss = (1.0f - p.E_route_loss);

        for (size_t i = 0; i < nvox; i++) {
            float x = clampf(curr.D[i] / p.D_ref, 0.0f, 1.0f);
            float eff_leak = p.E_global_leak * (1.0f + p.leak_k * x * x);
            eff_leak = clampf(eff_leak, 0.0f, 1.0f);

            float Ei = curr.E[i] * (1.0f - eff_leak);

            if (Ei <= 0.0f) {
                E_residual[i] = 0.0f;
                continue;
            }

            float c = conductivity(curr.D[i]) * (1.0f + R_boost[i]);
            float send = std::min(Ei * c, Ei);

            if (send <= 0.0f) {
                E_residual[i] = Ei;
                continue;
            }

            float delivered = send * loss;
            E_residual[i] = Ei - send;

            sent[i] = send;
            routed_sum_tick += send;
            routed_active_voxels_tick++;

            const float* w = &curr.W[i*6];
            for (int d = 0; d < 6; d++) {
                float frac = w[d];
                sent_dir[i*6 + d] = send * frac;

                int j = nbr[i*6 + d];
                if (j >= 0) next.E[(size_t)j] += delivered * frac;
            }
        }
    }

    static inline void sliding_max_1d_padded(
        const float* in, int L, int r, float* out
    ) {
        if (r <= 0) {
            std::copy(in, in + L, out);
            return;
        }

        const int w = 2*r + 1;
        const int T = L + 2*r;          // padded length
        const float NEG_INF = -1e30f;

        std::vector<int> dq;
        dq.reserve(w);

        auto get = [&](int t)->float {
            // padded index t maps to original index k=t-r
            int k = t - r;
            if (k < 0 || k >= L) return NEG_INF;
            return in[k];
        };

        int head = 0; // index into dq (manual pop_front without O(n))

        for (int t = 0; t < T; t++) {
            float v = get(t);

            // pop_back while last value <= v
            while ((int)dq.size() > head) {
                int back_idx = dq.back();
                if (get(back_idx) > v) break;
                dq.pop_back();
            }
            dq.push_back(t);

            // pop_front if out of window [t-w+1, t]
            int win_lo = t - w + 1;
            while ((int)dq.size() > head && dq[head] < win_lo) head++;

            // output when window fully formed: corresponds to center i = t - 2r
            int i = t - 2*r;
            if (i >= 0 && i < L) {
                out[i] = get(dq[head]);
            }

            // occasionally compact to avoid dq growing forever
            if (head > 1024) {
                dq.erase(dq.begin(), dq.begin() + head);
                head = 0;
            }
        }
    }

    void compute_sent_local_max_box(int r, std::vector<float>& out_local_max) {
        out_local_max.assign(nvox, 0.0f);

        // Shortcut: if radius covers entire grid, local max == global max everywhere
        int Rneed = std::max({p.nx-1, p.ny-1, p.nz-1});
        if (r >= Rneed) {
            float gmax = 0.0f;
            for (size_t i=0; i<nvox; i++) gmax = std::max(gmax, sent[i]);
            std::fill(out_local_max.begin(), out_local_max.end(), gmax);
            return;
        }

        // Pass 1: X lines
        for (int z=0; z<p.nz; z++) {
            for (int y=0; y<p.ny; y++) {
                const float* line = &sent[(z*p.ny + y)*p.nx];
                float* out = &tmpX[(z*p.ny + y)*p.nx];
                sliding_max_1d_padded(line, p.nx, r, out);
            }
        }

        // Pass 2: Y lines
        std::vector<float> inY(p.ny), outY(p.ny);
        for (int z=0; z<p.nz; z++) {
            for (int x=0; x<p.nx; x++) {
                // gather line along y
                for (int y=0; y<p.ny; y++) inY[y] = tmpX[idx(x,y,z)];
                sliding_max_1d_padded(inY.data(), p.ny, r, outY.data());
                for (int y=0; y<p.ny; y++) tmpY[idx(x,y,z)] = outY[y];
            }
        }

        // Pass 3: Z lines
        std::vector<float> inZ(p.nz), outZ(p.nz);
        for (int y=0; y<p.ny; y++) {
            for (int x=0; x<p.nx; x++) {
                for (int z=0; z<p.nz; z++) inZ[z] = tmpY[idx(x,y,z)];
                sliding_max_1d_padded(inZ.data(), p.nz, r, outZ.data());
                for (int z=0; z<p.nz; z++) out_local_max[idx(x,y,z)] = outZ[z];
            }
        }
    }

    void update_sent_tail() {
        static std::vector<float> local_max; // reuse
        int radius = p.sent_tail_radius;
        compute_sent_local_max_box(radius, local_max);

        for (size_t i=0; i<nvox; i++) {
            float lm = local_max[i];
            float& tail = sent_tail_local[i];

            if (lm > tail) {
                tail = tail + p.sent_tail_rise * (lm - tail);
            } else {
                tail *= p.sent_tail_decay;
            }

            if (tail < 1e-6f) tail = 1e-6f;
        }
    }

    // --------------------------------------------------------
    // REPAIR (scarce):
    // - Perpendicular neighbors only (geometry)
    // - Uses neighbor activity sent[j], not neighbor energy
    // - Neighbor must be below dead threshold (cannot repair if dead)
    // - Activity must exceed threshold (no dribble)
    // - Efficiency collapses with repairer damage
    // - Cost is superlinear (quadratic in amount) + activity multiplier
    // --------------------------------------------------------
    void apply_perpendicular_repair() {
        uint64_t active_repairers=0, eligible_repairers=0, eligible_active_repairers=0;

        float max_aj = 0.0f;

        // IMPORTANT: ensure these are zeroed for this tick (if they persist as members)
        // If they are std::vector<float>, do this at the start of each tick elsewhere:
        // std::fill(repair_delta.begin(), repair_delta.end(), 0.0f);
        // std::fill(repair_cost.begin(),  repair_cost.end(),  0.0f);

        for (size_t i=0; i<nvox; i++) {
            next.D[i] += p.D_activity_gain * sent[i];
            AxisInfo ax = dominant_axis(i);

            // Perp neighbors of TARGET voxel i (geometry)
            int j0=-1, j1=-1, j2=-1, j3=-1;
            if (ax.axis == XP) {
                j0 = nbr[i*6 + YP]; j1 = nbr[i*6 + YN];
                j2 = nbr[i*6 + ZP]; j3 = nbr[i*6 + ZN];
            } else if (ax.axis == YP) {
                j0 = nbr[i*6 + XP]; j1 = nbr[i*6 + XN];
                j2 = nbr[i*6 + ZP]; j3 = nbr[i*6 + ZN];
            } else {
                j0 = nbr[i*6 + XP]; j1 = nbr[i*6 + XN];
                j2 = nbr[i*6 + YP]; j3 = nbr[i*6 + YN];
            }

            auto try_repair_from = [&](int j) {
                if (j < 0) return;
                eligible_repairers++;

                float aj = sent[(size_t)j];
                max_aj = std::max(max_aj, aj);
                if (aj > 0.0f) active_repairers++;
                if (aj <= 0.0f) return;

                // Gate: spiky / rare repair
                float a_norm = sent[(size_t)j] / (curr.E[(size_t)j] + 1e-6f);
                a_norm = clampf(a_norm, 0.0f, 4.0f); // allow >1

                int x = int(i % p.nx);
                float tail_frac = p.repair_tail_frac;
                float eff_thresh = tail_frac * sent_tail_local[i];

                float Dgate = curr.D[i] / (curr.D[i] + 1e-2f); // ~0 when D small, ~1 when D big

                if (sent[(size_t)j] < eff_thresh) return;

                eligible_active_repairers++;

                float R = R_boost[i];

                // saturating kinetics
                float a = sent[i] / (curr.E[i] + 1e-6f);

                // center at threshold, smoothness controlled by slope
                float activity_gate = 1.0f / (1.0f + std::exp(-p.act_k * (a - p.act_thresh)));
                float P_available = curr.P[i];   // kinetics
                float dR = p.k_PR * P_available * activity_gate
                        / (1.0f + R / p.R_sat);

                // spend from next.P (which includes carryover + replenishment)
                dR = std::min(dR, next.P[i]);

                R_boost[i] += dR;
                next.P[i]  -= dR;

                // Noise per (tick, i, j) — stable
                float r11 = hash_to_f11(vh[i] ^ (vh[(size_t)j] << 1) ^ (tick * MIX_PAIR));
                float noise = 1.0f + p.repair_noise * r11;
                
                float amount = p.repair_strength * a_norm * noise * Dgate;

                // hard floor to avoid numeric extinction
                amount = std::max(amount, 1e-4f);

                // Superlinear scarcity (quadratic in amount)
                float cost = p.repair_energy_cost * (amount * amount);
                cost *= (1.0f + p.repair_cost_super_k * amount);
                cost *= (1.0f + p.repair_cost_activity_k * a_norm);

                // --- surface-area penalty ---
                float gD = local_D_gradient(i);
                float surf = 1.0f + p.repair_surface_k *
                    std::pow(gD, p.repair_surface_power);

                cost *= surf;

                float Ej = E_residual[j];   // ONLY leftover energy can be spent

                if (cost > Ej) return; 

                float R_spend = p.R_spend_k * amount;
                R_boost[i] = std::max(0.0f, R_boost[i] - R_spend);

                repair_delta[i] += amount;
                repair_cost[(size_t)j] += cost;
                E_residual[j] -= cost;
                repair_events_tick++;
            };

            try_repair_from(j0);
            try_repair_from(j1);
            try_repair_from(j2);
            try_repair_from(j3);
        }

        for (size_t i=0; i<nvox; i++) {
            float r = repair_delta[i];
            if (r > 0.0f) next.D[i] -= r;
        }
        for (size_t k = 0; k < nvox; k++) {
            next.E[k] += E_residual[k];
        }
    }


    // --------------------------------------------------------
    // WEIGHTS: evolve curr.W -> next.W (decay toward uniform + damage-linked noise)
    // --------------------------------------------------------
    void evolve_weights_curr_to_next() {
        for (size_t i = 0; i < nvox; i++) {
            float D = curr.D[i];
            float x = D / p.D_ref;
            if (x < 0.0f) x = 0.0f;
            if (x > 1.0f) x = 1.0f;

            float noise_amp = p.D_to_noise * D + p.porous_noise_boost * x * x;

            float w[6];
            const float* cw = &curr.W[i*6];
            for (int d = 0; d < 6; d++) {
                w[d] = lerpf(cw[d], 1.0f / 6.0f, p.W_decay);
            }

            for (int d = 0; d < 6; d++) {
                float r11 = hash_to_f11(
                    vh[i] ^ (tick * MIX_TICK) ^ (uint64_t(d) * MIX_DIR)
                );
                w[d] += noise_amp * r11;
            }

            normalize6(w);

            float* nw = &next.W[i*6];
            for (int d = 0; d < 6; d++) {
                nw[d] = w[d];
            }
        }
    }

    // --------------------------------------------------------
    // Diagnostics
    // --------------------------------------------------------
    void print_stats() {
        // ---- HARD GUARDS ----
        if (curr.D.size() != nvox) {
            std::cerr << "curr.D size mismatch\n";
            std::abort();
        }

        if (have_prev_snapshot) {
            if (D_prev.size() != curr.D.size()) {
                std::cerr << "D_prev size mismatch: "
                        << D_prev.size() << " vs " << curr.D.size() << "\n";
                std::abort();
            }
        }
        // Core + variance
        double Es=0.0, Ds=0.0, E2=0.0, D2=0.0, Rs=0.0;
        float Emax=0.0f, Dmax=0.0f;

        for (size_t i=0; i<nvox; i++) {
            float e = curr.E[i];
            float d = curr.D[i];
            float r = R_boost[i];
            Es += e; Ds += d; Rs += r;
            E2 += double(e) * double(e);
            D2 += double(d) * double(d);
            if (e > Emax) Emax = e;
            if (d > Dmax) Dmax = d;
        }

        double n = double(nvox);
        double Em = Es / n;
        double Dm = Ds / n;
        double VarE = (E2 / n) - Em*Em;
        double VarD = (D2 / n) - Dm*Dm;
        double corr_num = 0.0;
        double corr_den_curr = 0.0;
        double corr_den_prev = 0.0;
        if (have_prev_snapshot) {
            for (size_t i = 0; i < nvox; i++) {
                double dc = double(curr.D[i]) - Dm;
                double dp = double(D_prev[i]) - Dm_prev;  // see below
                corr_num += dc * dp;
                corr_den_curr += dc * dc;
                corr_den_prev += dp * dp;
            }
        }

        double v_all=0, v_rep=0, v_nrep=0;
        size_t n_all=0, n_rep=0, n_nrep=0;

        for (size_t i=0; i<nvox; i++) {
            double d = curr.D[i] - D_prev_tick[i];
            v_all += d*d; n_all++;

            if (repair_delta[i] > 0.0f) {
                v_rep += d*d; n_rep++;
            } else {
                v_nrep += d*d; n_nrep++;
            }
        }

        v_all  /= std::max<size_t>(1, n_all);
        v_rep  /= std::max<size_t>(1, n_rep);
        v_nrep /= std::max<size_t>(1, n_nrep);

        // Percentiles of D
        std::vector<float> tmpD(curr.D.begin(), curr.D.end());
        auto q_at = [&](double q)->float {
            size_t k = (size_t)std::floor(q * double(tmpD.size()-1));
            std::nth_element(tmpD.begin(), tmpD.begin()+k, tmpD.end());
            return tmpD[k];
        };
        float D_q10 = q_at(0.10);
        float D_q50 = q_at(0.50);
        float D_q95 = q_at(0.95);
        float D_q99 = q_at(0.99);

        std::vector<float> tmpS(sent.begin(), sent.end());

        auto qS_at = [&](double q)->float {
            size_t k = (size_t)std::floor(q * double(tmpS.size() - 1));
            std::nth_element(tmpS.begin(), tmpS.begin() + k, tmpS.end());
            return tmpS[k];
        };

        float sent_q95 = qS_at(0.95);
        float sent_q99 = qS_at(0.99);
        float sent_max = *std::max_element(tmpS.begin(), tmpS.end());
        sent_q95_cached = sent_q95;
        sent_q99_cached = sent_q99;

        // ------------------------
        // Repair Gating Index
        // ------------------------
        int repair_total = 0;
        int repair_high_flux = 0;
        int high_flux_total = 0;

        float flux_thresh = sent_q95_cached;  // or sent_q99_cached

        for (size_t i = 0; i < nvox; i++) {
            bool high_flux = sent[i] >= flux_thresh;
            bool repaired  = repair_delta[i] > 0.0f;

            if (repaired) repair_total++;
            if (high_flux) high_flux_total++;
            if (high_flux && repaired) repair_high_flux++;
        }

        double P_R = (double)repair_total / double(nvox);
        double P_R_given_S = high_flux_total > 0
            ? (double)repair_high_flux / double(high_flux_total)
            : 0.0;

        double RGI = (P_R > 1e-9) ? (P_R_given_S / P_R) : 0.0;

        // Junction density: voxels with >2 "significant" outgoing directions
        // (computed from curr.W; threshold chosen for interpretability)
        const float w_sig = 0.20f;
        int junctions = 0;
        for (size_t i=0; i<nvox; i++) {
            int deg = 0;
            const float* w = &curr.W[i*6];
            for (int d=0; d<6; d++) if (w[d] > w_sig) deg++;
            if (deg > 2) junctions++;
        }
        double junction_density = double(junctions) / n;

        // Transport metrics
        double mean_flux = (routed_active_voxels_tick > 0) ? (routed_sum_tick / routed_active_voxels_tick) : 0.0;
        double flux_to_storage = (Es > 0.0) ? (routed_sum_tick / Es) : 0.0;


        std::vector<size_t> idx(nvox);
        std::iota(idx.begin(), idx.end(), 0);

        std::nth_element(idx.begin(), idx.begin() + nvox/100, idx.end(),
            [&](size_t a, size_t b){ return sent[a] > sent[b]; });

        std::vector<uint8_t> top1(nvox, 0), top5(nvox, 0), top10(nvox, 0);

        for (size_t i = 0; i < nvox/100; i++) top1[idx[i]] = 1;
        for (size_t i = 0; i < nvox/20;  i++) top5[idx[i]] = 1;
        for (size_t i = 0; i < nvox/10;  i++) top10[idx[i]] = 1;
        auto overlap = [&](const std::vector<uint8_t>& a,
                        const std::vector<uint8_t>& b,
                        size_t K)->double {
            size_t inter = 0;
            for (size_t i = 0; i < nvox; i++)
                inter += (a[i] & b[i]);
            return double(inter) / double(K);
        };

        double ov1=0, ov5=0, ov10=0;
        if (have_top_prev) {
            ov1  = overlap(top1,  top1_prev,  nvox/100);
            ov5  = overlap(top5,  top5_prev,  nvox/20);
            ov10 = overlap(top10, top10_prev, nvox/10);
        }
        top1_prev = top1;
        top5_prev = top5;
        top10_prev = top10;
        have_top_prev = true;


        int max1=0, max5=0;
        double mean1=0, mean5=0;
        size_t c1=0, c5=0;

        for (size_t i=0; i<nvox; i++) {
            if (top1[i]) {
                top1_age[i]++;
                mean1 += top1_age[i];
                max1 = std::max(max1, top1_age[i]);
                c1++;
            } else {
                top1_age[i] = 0;
            }

            if (top5[i]) {
                top5_age[i]++;
                mean5 += top5_age[i];
                max5 = std::max(max5, top5_age[i]);
                c5++;
            } else {
                top5_age[i] = 0;
            }
        }

        mean1 /= std::max<size_t>(1, c1);
        mean5 /= std::max<size_t>(1, c5);

        double BSI = compute_BSI(curr.D, D_q50, nbr, nvox);
        double mean_R = 0.0;
        double mean_F = 0.0;

        for (size_t i = 0; i < nvox; i++) {
            if (repair_delta[i] > 0.0f) mean_R += 1.0;
            mean_F += sent[i];
        }
        mean_R /= double(nvox);
        mean_F /= double(nvox);
        rfc_buf.R.push_back(mean_R);
        rfc_buf.F.push_back(mean_F);

        if ((int)rfc_buf.R.size() > rfc_buf.maxlag + 4) {
            rfc_buf.R.erase(rfc_buf.R.begin());
            rfc_buf.F.erase(rfc_buf.F.begin());
        }
        double RFC = compute_RFC(rfc_buf);
        if (!fossil_frozen && BSI > 3.0) {   // crystal detected
            for (size_t i = 0; i < nvox; i++)
                fossil_C[i] = (curr.D[i] < D_q10) ? 1 : 0;
            fossil_frozen = true;
        }
        static std::vector<uint8_t> repaired_flag;
        repaired_flag.assign(nvox, 0);

        for (size_t i = 0; i < nvox; i++)
            if (repair_delta[i] > 0.0f)
                repaired_flag[i] = 1;
        double EMU = fossil_frozen ? compute_EMU(curr.D, fossil_C, repaired_flag, D_q50, nvox): 0.0;

        float Smax = 0, Dmax_local = 0, Pmax = 0, Rmax = 0, Emax_local = 0;
        for (size_t i=0; i<nvox; i++) {
            Smax = std::max(Smax, sent[i]);
            Dmax_local = std::max(Dmax_local, curr.D[i]);
            Pmax = std::max(Pmax, curr.P[i]);
            Rmax = std::max(Rmax, R_boost[i]);
            Emax_local = std::max(Emax_local, curr.E[i]);
        };

        // ------------------------
        // Repair Localization Index (entropy-based)
        // ------------------------
        double H = 0.0;
        double Rsum = 0.0;

        for (size_t i = 0; i < nvox; i++) {
            Rsum += repair_delta[i];
        }

        if (Rsum > 0.0) {
            for (size_t i = 0; i < nvox; i++) {
                double p = repair_delta[i] / Rsum;
                if (p > 1e-12) {
                    H -= p * std::log(p);
                }
            }
        }

        double Hmax = std::log(double(nvox));
        double RLI = (Hmax > 0.0) ? (1.0 - H / Hmax) : 0.0;

        // ------------------------
        // Structural Persistence Index (SPI)
        // ------------------------
        std::vector<size_t> ridx(nvox);
        std::iota(ridx.begin(), ridx.end(), 0);

        size_t K = nvox / 50;  // top 2% repair voxels
        std::nth_element(
            ridx.begin(),
            ridx.begin() + K,
            ridx.end(),
            [&](size_t a, size_t b) {
                return repair_delta[a] > repair_delta[b];
            }
        );

        std::vector<uint8_t> topR(nvox, 0);
        for (size_t i = 0; i < K; i++)
            if (repair_delta[ridx[i]] > 0.0f)
                topR[ridx[i]] = 1;

        double SPI = 0.0;
        if (have_topR_prev && K > 0) {
            size_t inter = 0;
            for (size_t i = 0; i < nvox; i++)
                inter += (topR[i] & topR_prev[i]);
            SPI = double(inter) / double(K);
        }

        topR_prev.swap(topR);
        have_topR_prev = true;

        std::cout
            << "tick=" << tick
            << " E_sum=" << std::fixed << std::setprecision(3) << Es
            << " E_max=" << Emax
            << " VarE=" << VarE
            << " D_sum=" << Ds
            << " D_ref=" << Dmax
            << " R_sum=" << Rs
            << " Rmax=" << Rmax
            << " Smax=" << Smax
            << " VarD=" << VarD
            //<< " D_q50=" << D_q50
            << " D_q95=" << D_q95
            << " D_q99=" << D_q99
            //<< " ov1=" << ov1
            //<< " ov5=" << ov5
            //<< " ov10=" << ov10
            //<< " age1_mean=" << mean1
            //<< " age1_max=" << max1
            //<< " age5_mean=" << mean5
            //<< " age5_max=" << max5
            //<< " Var_dD=" << v_all
            //<< " Var_dD_rep=" << v_rep
            //<< " Var_dD_nrep=" << v_nrep
            << " mean_flux=" << mean_flux
            << " flux/storage=" << flux_to_storage
            << " repair_events=" << repair_events_tick
            << " repair_eligible_frac=" << (double(repair_eligible_tick) / n)
            //<< " junction_density=" << junction_density
            << " BSI=" << BSI
            // << " RFC=" << RFC
            << " RGI=" << RGI
            << " RLI=" << RLI
            << " SPI=" << SPI
            // << " EMU=" << EMU
            << "\n";

        // ------------------------
        // CSV snapshot at 15k ticks
        // ------------------------
        if (!metrics_written && tick >= 15000) {
            append_metrics_csv(
                BSI,
                RFC,
                EMU,
                SPI,
                D_q50,
                D_q95,
                sent_q95_cached
            );
            metrics_written = true;
        }

        D_prev = curr.D;     // vector copy, but only every print_every
        Dm_prev = Dm;
        have_prev_snapshot = true;
        dump_fields();
    }
};

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    
    Params p;

    int steps = 15002;
    uint64_t seed = 15ull;
    std::cout << "BINARY BUILD ID: "
          << __DATE__ << " " << __TIME__
          << " | seed=" << seed
          << "\n";
    
    std::string load_path, save_path;
    int64_t save_at = -1;
    std::string param_file;
    std::vector<std::pair<std::string, std::string>> overrides;

    for (int i=1; i<argc; i++) {
        std::string a = argv[i];
        if (a == "--load" && i+1<argc) load_path = argv[++i];
        else if (a == "--save" && i+1<argc) save_path = argv[++i];
        else if (a == "--save_at" && i+1<argc) save_at = std::stoll(argv[++i]);
        else if (a == "--params" && i+1 < argc) param_file = argv[++i];
        else if (a == "--set" && i+1 < argc) {
            std::string kv = argv[++i];
            auto eq = kv.find('=');
            if (eq == std::string::npos) {
                std::cerr << "Bad --set format, expected key=value\n";
                return 1;
            }
            overrides.emplace_back(
                kv.substr(0, eq),
                kv.substr(eq + 1)
            );
        }
        else if (a == "--seed" && i+1 < argc) seed = argv[++i];
    }

    if (!param_file.empty()) {
        if (!p.load_from_file(param_file)) {
            std::cerr << "Failed to load params\n";
            return 1;
        }
    }

    for (auto& [k, v] : overrides) {
        if (!p.set_param(k, v)) {
            std::cerr << "Unknown parameter: " << k << "\n";
            return 1;
        }
    }

    World w(p, seed);

    if (!load_path.empty()) {
        if (!w.load_checkpoint(load_path)) {
            std::cerr << "Failed to load checkpoint\n";
            return 1;
        }
    }
    for (int t=0; t<steps; t++) {
        w.step();

        if (save_at >= 0 && (int64_t)w.tick == save_at) {
            if (!w.save_checkpoint(save_path.empty() ? "checkpoint.bin" : save_path)) {
                std::cerr << "Failed to save checkpoint\n";
                return 1;
            }
            std::cout << "Saved checkpoint at tick=" << w.tick << "\n";
            return 0; // exit as you requested
        }

        if ((t % p.print_every) == 0) w.print_stats();
    }
}
// example:
// voxel 200000000 12345 --save_at 3500 --save ckpt.bin
// voxel 200000000 12345 --load ckpt.bin