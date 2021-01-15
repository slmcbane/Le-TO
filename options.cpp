#include "options.hpp"

#include <cassert>
#include <fstream>

toml::table parse_options_file(const char *path) { return toml::parse_file(path); }

namespace
{

double retrieve_double(const toml::node *option)
{
    if (!option->is_number())
    {
        throw OptionException("Expected a numeric options\n");
    }
    else if (option->is_floating_point())
    {
        return option->as_floating_point()->get();
    }
    else
    {
        assert(option->is_integer());
        return option->as_integer()->get();
    }
}

} // namespace

constexpr inline const char *material_option_names[] = {
    "epsilon_constitutive", "epsilon_stress",        "simp_exponent", "stress_exponent",
    "youngs_modulus",       "youngs_modulus_stress", "poissons_ratio"};

void set_value(MaterialOptions &mat_options, const char *what, double x)
{
    switch (what[0])
    {
    case 'e':
        if (what[8] == 'c')
        {
            mat_options.epsilon_constitutive = x;
        }
        else
        {
            mat_options.epsilon_stress = x;
        }
        break;
    case 's':
        if (what[1] == 'i')
        {
            mat_options.simp_exponent = x;
        }
        else
        {
            mat_options.stress_exponent = x;
        }
        break;
    case 'y':
        if (what[14] == '\0')
        {
            mat_options.youngs_modulus = x;
        }
        else
        {
            mat_options.youngs_modulus_stress = x;
        }
        break;
    case 'p':
        mat_options.poissons_ratio = x;
        break;
    default:;
    }
}

void parse_material_options(const toml::table &options, MaterialOptions &mat_options)
{
    const toml::node *material_table = options.get("Material");
    if (material_table == nullptr)
    {
        throw OptionException("Requested material options, but no [Material] heading in options\n");
    }
    else if (!material_table->is_table())
    {
        throw OptionException("Got Material as something other than table in TOML\n");
    }

    for (const char *option_name : material_option_names)
    {
        const toml::node *option = material_table->as_table()->get(option_name);
        if (option != nullptr)
        {
            if (!option->is_number())
            {
                throw OptionException(
                    "Received unexpected non-numeric option while parsing material properties\n");
            }
            else if (option->is_floating_point())
            {
                set_value(mat_options, option_name, option->as_floating_point()->get());
            }
            else
            {
                set_value(mat_options, option_name, option->as_integer()->get());
            }
        }
    }
}

void parse_aggregation_options(const toml::table &options, AggregationOptions &agg_options)
{
    const toml::node *agg_table = options.get("Aggregation");
    if (agg_table == nullptr)
    {
        throw OptionException("Requested aggregation options, but no [Aggregation] heading in options\n");
    }
    else if (!agg_table->is_table())
    {
        throw OptionException("Got Aggregation as something other than table in TOML\n");
    }

    const toml::node *option = agg_table->as_table()->get("num_aggregation_regions");
    if (option != nullptr)
    {
        if (!option->is_integer())
        {
            throw OptionException("num_aggregation_regions should be an integer\n");
        }
        agg_options.num_aggregation_regions = option->as_integer()->get();
    }

    option = agg_table->as_table()->get("aggregation_multiplier");
    if (option != nullptr)
    {
        if (!option->is_number())
        {
            throw OptionException("aggregation_multiplier should be a number\n");
        }
        agg_options.aggregation_multiplier = retrieve_double(option);
    }
}

constexpr inline const char *opt_option_names[] = {
    "max_iters",
    "absolute_tol",
    "acceptable_tol",
    "acceptable_iters",
    "accept_every_trial_step",
    "mass_scaling",
    "compliance_scaling",
    "compliance_limit",
    "stress_scaling",
    "stress_limit",
    "mass_limit",
    "mean_change_iters",
    "mean_change_tol",
    "verbosity_level",
    "xtol_rel",
    "xtol_abs",
    "ftol_rel",
    "ftol_abs",
    "maxtime",
    "optimization_type",
    "reassign_regions_interval",
    "stress_alpha"};

template <class T>
void set_value(OptimizationOptions &options, const char *what, T value)
{
    switch (what[0])
    {
    case 'a':
        if (what[1] == 'b')
        {
            if (!std::is_same_v<T, double>)
            {
                throw OptionException("absolute_tol should be a floating point option\n");
            }
            options.absolute_tol = value;
            break;
        }
        else if (what[6] == '_')
        {
            if (!std::is_same_v<T, bool>)
            {
                throw OptionException("accept_every_trial_step should be a boolean option\n");
            }
            options.accept_every_trial_step = value;
            break;
        }
        else if (what[11] == 't')
        {
            if (!std::is_same_v<T, double>)
            {
                throw OptionException("acceptable_tol should be a floating point option\n");
            }
            options.acceptable_tol = value;
            break;
        }
        else
        {
            options.acceptable_iters = value;
            break;
        }
    case 'm':
        if (what[1] == 'a')
        {
            if (what[2] == 'x')
            {
                if (what[3] == '_')
                {
                    options.max_iters = value;
                }
                else
                {
                    assert(what[3] == 't');
                    options.maxtime = value;
                }
            }
            else
            {
                assert(what[2] == 's');
                if (what[5] == 's')
                {
                    options.mass_scaling = value;
                }
                else
                {
                    assert(what[5] == 'l');
                    options.mass_limit = value;
                }
            }
        }
        else if (what[12] == 'i')
        {
            options.mean_change_iters = value;
        }
        else
        {
            options.mean_change_tol = value;
        }
        break;
    case 's':
        if (what[7] == 'l')
        {
            options.stress_limit = value;
        }
        else if (what[7] == 's')
        {
            options.stress_scaling = value;
        }
        else
        {
            assert(what[7] == 'a');
            options.stress_alpha = value;
        }
        break;
    case 'v':
        options.verbosity_level = value;
        break;
    case 'x':
        if (what[5] == 'r')
        {
            options.xtol_rel = value;
        }
        else
        {
            assert(what[5] == 'a');
            options.xtol_abs = value;
        }
        break;
    case 'f':
        if (what[5] == 'r')
        {
            options.ftol_rel = value;
        }
        else
        {
            assert(what[5] == 'a');
            options.ftol_abs = value;
        }
        break;
    case 'c':
        if (what[11] == 's')
        {
            options.compliance_scaling = value;
        }
        else
        {
            assert(what[11] == 'l');
            options.compliance_limit = value;
        }
        break;
    case 'r':
        options.reassign_regions_interval = value;
        break;
    default:;
    }
}

void parse_optimization_options(const toml::table &options, OptimizationOptions &opt_options)
{
    const toml::node *opt_table = options.get("Optimization");
    if (opt_table == nullptr)
    {
        throw OptionException("Requested optimization options, but no [Optimization] heading in options\n");
    }
    else if (!opt_table->is_table())
    {
        throw OptionException("Got Optimization as something other than table in TOML\n");
    }

    for (const char *name : opt_option_names)
    {
        const toml::node *option = opt_table->as_table()->get(name);
        if (option == nullptr)
        {
            continue;
        }

        if (option->is_integer())
        {
            set_value(opt_options, name, option->as_integer()->get());
        }
        else if (option->is_floating_point())
        {
            set_value(opt_options, name, option->as_floating_point()->get());
        }
        else if (option->is_boolean())
        {
            set_value(opt_options, name, option->as_boolean()->get());
        }
        else
        {
            throw OptionException("Got unrecognized type for option in parse_optimization_options\n");
        }
    }
}

constexpr const char *solver_option_names[] = {"maxiters", "tol"};

template <class T>
void set_value(SolverOptions &options, const char *what, T value)
{
    static_assert(std::is_arithmetic_v<T>);
    if (what[0] == 'm')
    {
        options.maxiters = value;
    }
    else
    {
        if (!std::is_same_v<T, double>)
        {
            throw OptionException("tol should be a floating point option\n");
        }
        options.tol = value;
    }
}

void parse_solver_options(const toml::table &options, SolverOptions &sol_options)
{
    const toml::node *sol_table = options.get("Solver");
    if (!sol_table)
    {
        throw OptionException("Requested solver options, but no [Solver] heading in options\n");
    }
    else if (!sol_table->is_table())
    {
        throw OptionException("Got Solver as something other than table in options\n");
    }

    for (const char *name : solver_option_names)
    {
        const toml::node *option = sol_table->as_table()->get(name);
        if (option == nullptr)
        {
            continue;
        }

        if (option->is_integer())
        {
            set_value(sol_options, name, option->as_integer()->get());
        }
        else if (option->is_floating_point())
        {
            set_value(sol_options, name, option->as_floating_point()->get());
        }
        else
        {
            throw OptionException("Got unexpected type for option in parse_solver_options\n");
        }
    }
}

constexpr const char *model_reduction_option_names[] = {
    "regularization", "num_snapshots", "tol", "basis_size"};

template <class T>
void set_value(ModelReductionOptions &options, const char *what, T value)
{
    static_assert(std::is_arithmetic_v<T>);
    switch (what[0])
    {
    case 'r':
        options.regularization = value;
        break;
    case 'n':
        options.num_snapshots = value;
        break;
    case 't':
        options.tol = value;
        break;
    case 'b':
        options.basis_size = value;
        break;
    default:;
    }
}

void parse_model_reduction_options(const toml::table &options, ModelReductionOptions &mr_options)
{
    const toml::node *mr_table = options.get("ModelReduction");
    if (mr_table == nullptr)
    {
        throw OptionException(
            "Requested model reduction options, but no [ModelReduction] heading in options\n");
    }
    else if (!mr_table->is_table())
    {
        throw OptionException("Got ModelReduction as something other than table in TOML\n");
    }

    for (const char *name : model_reduction_option_names)
    {
        const toml::node *option = mr_table->as_table()->get(name);
        if (option == nullptr)
        {
            continue;
        }

        if (option->is_integer())
        {
            set_value(mr_options, name, option->as_integer()->get());
        }
        else if (option->is_floating_point())
        {
            set_value(mr_options, name, option->as_floating_point()->get());
        }
        else
        {
            throw OptionException("Got unexpected type for option in parse_model_reduction_options\n");
        }
    }
}

constexpr const char *amg_option_names[] = {
    "strong_threshold",
    "print_level",
    "max_levels",
    "max_coarse_size",
    "min_coarse_size",
    "relax_type",
    "down_relax_type",
    "up_relax_type",
    "coarse_relax_type",
    "smooth_type",
    "smooth_num_levels",
    "smooth_num_sweeps",
    "euclid_file",
    "coarsen_type",
    "interp_type",
    "parasails_threshold",
    "parasails_level",
    "parasails_filter",
    "agg_levels",
    "agg_num_paths",
    "agg_interp_type",
    "agg_trunc_factor",
    "agg_p12_trunc_factor",
    "agg_pmax_elements",
    "agg_p12max_elements",
    "ilu_level",
    "ilu_type",
    "ilu_row_nnz",
    "ilu_drop_threshold",
    "ilu_maxiter"};

template <class T>
std::enable_if_t<std::is_arithmetic_v<T>, void>
set_value(BoomerAMGOptions &options, const char *what, T value)
{
    switch (what[0])
    {
    case 's':
        if (what[1] == 't')
        {
            options.strong_threshold = value;
        }
        else if (what[7] == 't')
        {
            options.smooth_type = value;
        }
        else if (what[11] == 'l')
        {
            options.smooth_num_levels = value;
        }
        else
        {
            assert(what[11] == 's');
            options.smooth_num_sweeps = value;
        }
        break;
    case 'p':
        if (what[1] == 'r')
        {
            options.print_level = value;
        }
        else
        {
            assert(what[1] == 'a');
            switch (what[10])
            {
            case 't':
                options.parasails_threshold = value;
                break;
            case 'l':
                if constexpr (!std::is_integral_v<T>)
                {
                    throw OptionException("parasails_level should be an integer value\n");
                }
                options.parasails_level = value;
                break;
            case 'f':
                options.parasails_filter = value;
                break;
            default:
                assert(false && "Unrecognized option hit\n");
            }
        }
        break;
    case 'm':
        if (what[1] == 'i')
        {
            options.min_coarse_size = value;
        }
        else if (what[4] == 'c')
        {
            options.max_coarse_size = value;
        }
        else
        {
            assert(what[4] == 'l');
            options.max_levels = value;
        }
        break;
    case 'r':
        options.relax_type = value;
        break;
    case 'd':
        options.down_relax_type = value;
        break;
    case 'u':
        options.up_relax_type = value;
        break;
    case 'c':
        if (what[6] == '_')
        {
            options.coarse_relax_type = value;
        }
        else
        {
            assert(what[6] == 'n');
            options.coarsen_type = value;
        }
        break;
    case 'i':
        if (what[1] == 'n')
        {
            options.interp_type = value;
        }
        else if (what[4] == 'l')
        {
            options.ilu_level = value;
        }
        else if (what[4] == 't')
        {
            options.ilu_type = value;
        }
        else if (what[4] == 'r')
        {
            options.ilu_row_nnz = value;
        }
        else if (what[4] == 'm')
        {
            options.ilu_maxiter = value;
        }
        else
        {
            assert(what[4] == 'd');
            options.ilu_drop_threshold = value;
        }
        break;
    case 'a': // aggressive coarsening options.
        switch (what[4])
        {
        case 'p':
            if (what[5] == '1')
            {
                if (what[7] == 'm')
                {
                    options.agg_p12max_elements = value;
                }
                else
                {
                    assert(what[7] == '_');
                    options.agg_p12_trunc_factor = value;
                }
            }
            else
            {
                assert(what[5] == 'm');
                options.agg_pmax_elements = value;
            }
            break;
        case 'l':
            options.agg_levels = value;
            break;
        case 'n':
            options.agg_num_paths = value;
            break;
        case 'i':
            options.agg_interp_type = value;
            break;
        case 't':
            options.agg_trunc_factor = value;
            break;
        default:
            throw OptionException("Fell through switch under case 'a'\n");
        }
    default:;
    }
}

template <class T>
std::enable_if_t<std::is_same_v<T, std::string>, void>
set_value(BoomerAMGOptions &options, const char *what, const T &value)
{
    assert(what[0] == 'e' && "Passed string value other than euclid_file");
    options.euclid_file = value;
}

void parse_amg_options(const toml::table &options, BoomerAMGOptions &amg_options)
{
    const toml::node *amg_table = options.get("BoomerAMG");
    if (amg_table == nullptr)
    {
        throw OptionException("Requested AMG options, but no [BoomerAMG] heading found\n");
    }
    else if (!amg_table->is_table())
    {
        throw OptionException("Got BoomerAMG as something other than table in TOML\n");
    }

    for (const char *name : amg_option_names)
    {
        const toml::node *option = amg_table->as_table()->get(name);
        if (option == nullptr)
        {
            continue;
        }

        if (name[0] == 'e')
        {
            if (!option->is_string())
            {
                throw OptionException("Got euclid_file as something other than string\n");
            }
            set_value(amg_options, name, option->as_string()->get());
        }
        else if (option->is_integer())
        {
            set_value(amg_options, name, option->as_integer()->get());
        }
        else if (option->is_floating_point())
        {
            set_value(amg_options, name, option->as_floating_point()->get());
        }
        else
        {
            throw OptionException("Got unexpected type for option in parse_model_reduction_options\n");
        }
    }
}
