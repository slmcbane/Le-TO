#ifndef FORWARD_MODEL_HPP
#define FORWARD_MODEL_HPP

#include "2d.hpp"
#include "density-filter.hpp"
#include "ersatz_stiffness.hpp"

#include <Eigen/Sparse>

template <class Mesh>
struct ModelInfo
{
    using element_t = decltype(Elasticity::TwoD::instantiate_element(std::declval<const Mesh &>(), 0));
    using el_stiffness_matrix_t =
        decltype(Elasticity::TwoD::element_stiffness_matrix(std::declval<element_t>(), 0, 0));
    using stiffness_matrix_t = Elasticity::TwoD::StiffnessType<Mesh>;

    // We need the mesh data structure around.
    Mesh mesh;

    // Interpolation for the stiffness.
    ErsatzStiffness interp;

    // Density filter.
    std::vector<DensityFilter> filter;

    // Reference values of the element stiffness matrices.
    std::vector<el_stiffness_matrix_t> ref_stiffness_matrices;

    // Segments of the boundary with a homogeneous Dirichlet BC.
    std::vector<size_t> homogeneous_boundaries;

    // Store filtered values of the density variable.
    std::vector<double> rho_filt;

    // Storage for the forward stiffness matrix.
    stiffness_matrix_t forward_stiffness;
    Eigen::SparseMatrix<double> forward_stiffness_eigen;
    // Factorization of the stiffness.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorized;

    // The right hand side and displacement vectors.
    Eigen::VectorXd nodal_forcing;
    Eigen::VectorXd displacement;

    ModelInfo(Mesh &&m, ErsatzStiffness i, std::vector<DensityFilter> &&filt, std::vector<size_t> &&hom,
        Eigen::VectorXd &&forcing)
        : mesh(std::move(m)), interp(i), filter(std::move(filt)), homogeneous_boundaries(std::move(hom)),
          forward_stiffness(2 * mesh.num_nodes()), nodal_forcing(std::move(forcing)),
          displacement(Eigen::VectorXd::Zero(nodal_forcing.size()))
    {
        assert(displacement.size() == 2 * mesh.num_nodes());
    }
};

template <class Mesh>
void update_forward_stiffness(ModelInfo<Mesh> &minfo, std::size_t eli, bool init = false);

template <class Mesh>
void construct_eigen_stiffness(
    Eigen::SparseMatrix<double> &Ke, const typename ModelInfo<Mesh>::stiffness_matrix_t &K);

template <class Mesh>
void initialize_model_info(ModelInfo<Mesh> &minfo, double lambda, double mu)
{
    const auto &mesh = minfo.mesh;
    minfo.ref_stiffness_matrices.reserve(mesh.num_elements());
    minfo.rho_filt.resize(mesh.num_elements(), 1.0);

    // Compute the reference stiffness matrices, and also the structure of the
    // sparse global stiffness matrix.
    for (std::size_t eli = 0; eli < mesh.num_elements(); ++eli)
    {
        minfo.ref_stiffness_matrices.emplace_back(Elasticity::TwoD::element_stiffness_matrix(
            Elasticity::TwoD::instantiate_element(mesh, eli), lambda, mu));

        update_forward_stiffness(minfo, eli, true);
    }

    for (std::size_t which : minfo.homogeneous_boundaries)
    {
        Elasticity::TwoD::impose_homogeneous_condition(mesh, minfo.forward_stiffness, which);
    }
    construct_eigen_stiffness(minfo.forward_stiffness_eigen, minfo.forward_stiffness);
    minfo.factorized.analyzePattern(minfo.forward_stiffness_eigen);
    
    return minfo;
}

template <class Mesh, class Vec>
void update_model_info(ModelInfo<Mesh> &minfo, const Vec &rho)
{
    filter_densities(minfo.rho_filt, rho, minfo.filter);
    minfo.forward_stiffness.reset();
    for (std::size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        update_forward_stiffness(minfo, eli);
    }
    for (std::size_t which : minfo.homogeneous_boundaries)
    {
        Elasticity::TwoD::impose_homogeneous_condition(minfo.mesh, minfo.forward_stiffness, which);
    }
    construct_eigen_stiffness(minfo.forward_stiffness_eigen, minfo.forward_stiffness);
}

template <class Mesh>
void update_forward_stiffness(ModelInfo<Mesh> &minfo, std::size_t eli, bool init)
{
    const auto &nn = minfo.mesh.element(eli).node_numbers();
    const auto &Kel = minfo.ref_stiffness_matrices.at(eli);
    assert(nn.size() == 2 * Kel.rows());
    assert(Kel.rows() == Kel.cols());

    double scaling = init ? 1.0 : minfo.interp(minfo.rho_filt[eli]);
    auto &K = minfo.forward_stiffness;
    for (std::size_t i = 0; i < nn.size(); ++i)
    {
        for (std::size_t j = i; j < nn.size(); ++j)
        {
            K.insert_entry(2 * nn[i], 2 * nn[j], scaling * Kel(2 * i, 2 * j));
            K.insert_entry(2 * nn[i], 2 * nn[j] + 1, scaling * Kel(2 * i, 2 * j + 1));
            if (i != j)
            {
                K.insert_entry(2 * nn[i] + 1, 2 * nn[j], scaling * Kel(2 * i + 1, 2 * j));
            }
            K.insert_entry(2 * nn[i] + 1, 2 * nn[j] + 1, scaling * Kel(2 * i + 1, 2 * j + 1));
        }
    }
}

template <class Mesh>
void construct_eigen_stiffness(
    Eigen::SparseMatrix<double> &Ke, const typename ModelInfo<Mesh>::stiffness_matrix_t &K)
{
    std::vector<Eigen::Triplet<double>> triplets;
    for (std::size_t i = 0; i < K.num_rows(); ++i)
    {
        for (auto [j, v] : K.row(i))
        {
            triplets.emplace_back(i, j, v);
            if (j != i)
            {
                triplets.emplace_back(j, i, v);
            }
        }
    }
    Ke.setFromTriplets(triplets.begin(), triplets.end());
    Ke.makeCompressed();
}

#endif // FORWARD_MODEL_HPP
