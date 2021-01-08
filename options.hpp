#ifndef OPTIONS_HPP
#define OPTIONS_HPP

/*
 * The options defined in these structs have a 1-1 correspondence with options that
 * are recognized in a .toml file containing the options. Section headers are what you'd
 * expect from the struct name - e.g. [Material], [Aggregation], [Optimization], etc.
 *
 * Options with default values are set in the default constructors defined inline here.
 */

#include "toml.hpp"
#include <exception>
#include <optional>

toml::table parse_options_file(const char *path);

struct OptionException : public std::exception
{
    OptionException(const char *msg) : m_msg{msg} {}
    const char *m_msg;

    const char *what() const noexcept { return m_msg; }
};

struct MaterialOptions
{
    std::optional<double> epsilon_constitutive;
    std::optional<double> epsilon_stress;
    std::optional<double> simp_exponent;
    std::optional<double> stress_exponent;

    std::optional<double> youngs_modulus;
    std::optional<double> youngs_modulus_stress;
    std::optional<double> poissons_ratio;

    MaterialOptions() : simp_exponent{3}, stress_exponent{0.5} {}
};

struct AggregationOptions
{
    std::optional<int> num_aggregation_regions;
    std::optional<double> aggregation_multiplier;
    std::optional<std::vector<std::vector<long>>> region_definitions;

    AggregationOptions() : num_aggregation_regions{1} {}
};

enum OptimizationType
{
    MWS,
    MCW,
    MSW,
    MCWS,
    MWCS
};

struct OptimizationOptions
{
    std::optional<int> max_iters;
    std::optional<double> absolute_tol;
    std::optional<double> acceptable_tol;
    std::optional<int> acceptable_iters;
    std::optional<bool> accept_every_trial_step;
    std::optional<double> mass_scaling;
    std::optional<double> compliance_scaling;
    std::optional<double> compliance_limit;
    std::optional<double> stress_scaling;
    std::optional<double> stress_limit;
    std::optional<double> mass_limit;
    std::optional<int> mean_change_iters;
    std::optional<double> mean_change_tol;
    std::optional<int> verbosity_level;

    std::optional<double> xtol_rel;
    std::optional<double> xtol_abs;
    std::optional<double> ftol_rel;
    std::optional<double> ftol_abs;
    std::optional<double> maxtime;

    OptimizationOptions() {}
};

struct SolverOptions
{
    std::optional<int> maxiters;
    std::optional<double> tol;

    SolverOptions() {}
};

struct ModelReductionOptions
{
    std::optional<double> regularization;
    std::optional<int> num_snapshots;
    std::optional<double> tol;
    std::optional<int> basis_size;

    ModelReductionOptions() {}
};

struct BoomerAMGOptions
{
    std::optional<double> strong_threshold;
    std::optional<int> print_level;
    std::optional<int> max_levels;
    std::optional<int> max_coarse_size;
    std::optional<int> min_coarse_size;
    std::optional<int> relax_type;
    std::optional<int> down_relax_type;
    std::optional<int> up_relax_type;
    std::optional<int> coarse_relax_type;
    std::optional<int> smooth_type;
    std::optional<int> smooth_num_levels;
    std::optional<int> smooth_num_sweeps;
    std::optional<std::string> euclid_file;
    std::optional<int> coarsen_type;
    std::optional<int> interp_type;
    std::optional<double> parasails_threshold;
    std::optional<int> parasails_level;
    std::optional<double> parasails_filter;

    std::optional<int> agg_levels;
    std::optional<int> agg_num_paths;
    std::optional<int> agg_interp_type;
    std::optional<double> agg_trunc_factor;
    std::optional<double> agg_p12_trunc_factor;
    std::optional<int> agg_pmax_elements;
    std::optional<int> agg_p12max_elements;

    std::optional<int> ilu_level;
    std::optional<int> ilu_type;
    std::optional<int> ilu_row_nnz;
    std::optional<int> ilu_maxiter;
    std::optional<double> ilu_drop_threshold;

    BoomerAMGOptions() {}
};

void parse_material_options(const toml::table &options, MaterialOptions &mat_options);
void parse_aggregation_options(const toml::table &options, AggregationOptions &agg_options);
void parse_optimization_options(const toml::table &options, OptimizationOptions &opt_options);
void parse_solver_options(const toml::table &options, SolverOptions &sol_options);
void parse_model_reduction_options(const toml::table &options, ModelReductionOptions &mr_options);
void parse_amg_options(const toml::table &options, BoomerAMGOptions &amg_options);

template <class T>
T try_to_get_as(const toml::table &options, const char *name)
{
    std::optional<T> val = options[name].value<T>();
    if (!val)
    {
        throw OptionException("Failed to retrieve option as requested type\n");
    }
    return *val;
}

#endif // OPTIONS_HPP
