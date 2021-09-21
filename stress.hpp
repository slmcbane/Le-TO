#ifndef STRESS_COMPUTATIONS_HPP
#define STRESS_COMPUTATIONS_HPP

#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "forward_model.hpp"
#include "von_mises.hpp"

void cell_centered_stress(Eigen::VectorXd &dest, const ModelInfoVariant &minfo, double lambda, double mu);

Eigen::VectorXd averaged_nodal_stress(
    const ModelInfoVariant &, const std::vector<int> &nadjacent, const Eigen::VectorXd &rho, double lambda,
    double mu);

/*
 * Compute element-wise maximum stress over quadrature points (from 8th-order quadrature rule)
 * and nodes.
 */
Eigen::VectorXd
elemental_max_stress(const ModelInfoVariant &, const Eigen::VectorXd &rho, double lambda, double mu);

struct AggregationRegions
{
    std::size_t n;
    std::vector<std::size_t> assignments;

    AggregationRegions(std::size_t num_elements) : n{1} { assignments.resize(num_elements, 0); }

    AggregationRegions() = default;
};

struct StressCriterionDefinition
{
    ErsatzStiffness stiffness_interp;
    AggregationRegions agg_regions;
    double p, stress_limit, alpha;
};

/*
 * Compute the stress aggregates as well as the cell centered stress values
 * according to the definition of stress regions.
 */
void pnorm_stress_aggregates(
    Eigen::VectorXd &aggs, Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def,
    const ModelInfoVariant &minfo, double lambda, double mu);

/*
 * Compute stress aggregates with the KS functional using given aggregation regions.
 */
void ks_stress_aggregates(
    Eigen::VectorXd &aggs, const StressCriterionDefinition &def, const ModelInfoVariant &minfo,
    double lambda, double mu);

void ks_aggs_with_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    const StressCriterionDefinition &def, const ModelInfoVariant &minfo, double lambda, double mu,
    Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2);

AggregationRegions assign_agg_regions(const Eigen::VectorXd &cc_stress, std::size_t n);

/*
 * The same as pnorm_stress_aggregates, but also computes the Jacobian of the constraints
 * in a single pass over the domain (with linear solve).
 */
void pnorm_aggs_with_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def, const ModelInfoVariant &minfo,
    double lambda, double mu, Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2);

/*
 * Estimate the normalization alpha for K-S aggregation via eq. (8) from
 * Kennedy and Hicken. The returned alpha is frac * upper bound from that
 * equation. p is multiplier in the K-S functional.
 */
double estimate_ks_alpha(
    const ModelInfoVariant &minfo, double lambda, double mu, double p, double stress_limit, double frac);

/*
 * Estimate maximum stress by evaluating at all quadrature points
 * (as used in evaluating the K-S aggregates) and nodes.
 */
double estimate_max_stress(const ModelInfoVariant &minfo, double lambda, double mu);

#endif // STRESS_COMPUTATIONS_HPP
