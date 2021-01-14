#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "ersatz_stiffness.hpp"
#include "forward_model.hpp"
#include "stress.hpp"
#include "tensor_product.hpp"

#include <optional>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

class Evaluator
{
  public:
    Evaluator(
        const std::string &mesh_file, int order, Eigen::Vector2d force, ErsatzStiffness interp,
        double filt_radius, double lambda, double mu)
        : m_minfo(construct_model_info(mesh_file, order, force, interp, filt_radius, lambda, mu)),
          parameter_set{false}, solved_forward{false}, compliance_computed{false},
          compliance_gradient_computed{false}, cc_stress_computed{false}, aggregates_computed{false},
          aggj_computed{false}, lambda{lambda}, mu{mu}
    {
    }

    const Eigen::VectorXd &displacement();

    void set_parameter(const double *rho);

    double compliance();

    const Eigen::VectorXd &compliance_gradient();

    const ModelInfoVariant &model_info() const { return *m_minfo; }

    const Eigen::VectorXd &cell_centered_stress();

    const Eigen::VectorXd &stress_aggregates();

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &stress_agg_jacobian();

    void set_stress_criterion(StressCriterionDefinition &&criterion)
    {
        stress_criterion = std::move(criterion);
    }

  private:
    std::unique_ptr<ModelInfoVariant> m_minfo;

    bool parameter_set;
    bool solved_forward;
    bool compliance_computed;
    bool compliance_gradient_computed;
    bool cc_stress_computed;
    bool aggregates_computed;
    bool aggj_computed;

    void solve_forward();
    void compute_compliance();
    void compute_compliance_gradient();

    void compute_cc_stress();
    void compute_aggregates();
    void compute_aggregates_and_jac();

    double compliance_value;
    Eigen::VectorXd compliance_gradient_value;

    std::optional<StressCriterionDefinition> stress_criterion;

    Eigen::VectorXd cc_stress_value;
    Eigen::VectorXd aggregate_values;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> agg_jacobian;
    Eigen::MatrixXd workspace, workspace2;

    double lambda, mu;

    void check_stress_defined()
    {
        if (!stress_criterion)
        {
            fmt::print(
                stderr, "Use set_stress_criterion to define the stress aggregation scheme before using "
                        "aggregation functions\n");
            exit(3);
        }
    }
};

#endif // EVALUATOR_HPP
