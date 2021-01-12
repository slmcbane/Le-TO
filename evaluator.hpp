#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "ersatz_stiffness.hpp"
#include "forward_model.hpp"
#include "tensor_product.hpp"

class Evaluator
{
  public:
    Evaluator(
        const std::string &mesh_file, int order, Eigen::Vector2d force, ErsatzStiffness interp,
        double filt_radius, double lambda, double mu)
        : m_minfo(construct_model_info(mesh_file, order, force, interp, filt_radius, lambda, mu)),
          parameter_set{false}, solved_forward{false}, compliance_computed{false},
          compliance_gradient_computed{false}
    {
    }

    const Eigen::VectorXd &displacement();

    void set_parameter(const double *rho);

    double compliance();

    const Eigen::VectorXd &compliance_gradient();

    const ModelInfoVariant &model_info() const { return *m_minfo; }

  private:
    std::unique_ptr<ModelInfoVariant> m_minfo;

    bool parameter_set;
    bool solved_forward;
    bool compliance_computed;
    bool compliance_gradient_computed;

    void solve_forward();
    void compute_compliance();
    void compute_compliance_gradient();

    double compliance_value;
    Eigen::VectorXd compliance_gradient_value;
};

#endif // EVALUATOR_HPP
