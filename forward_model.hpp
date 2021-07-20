#ifndef FORWARD_MODEL_HPP
#define FORWARD_MODEL_HPP

#include "2d.hpp"
#include "boundaries.hpp"
#include "density-filter.hpp"
#include "ersatz_stiffness.hpp"
#include "read_mesh.hpp"

#include <Eigen/Sparse>

#include <memory>

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
    Eigen::VectorXd rho_filt;

    // Storage for the forward stiffness matrix.
    stiffness_matrix_t forward_stiffness;
    Eigen::SparseMatrix<double> forward_stiffness_eigen;
    // Factorization of the stiffness.
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> factorized;

    // The right hand side and displacement vectors.
    Eigen::VectorXd nodal_forcing;
    Eigen::VectorXd displacement;

    ModelInfo(
        Mesh &&m, ErsatzStiffness i, std::vector<DensityFilter> &&filt, std::vector<size_t> &&hom,
        Eigen::VectorXd &&forcing)
        : mesh(std::move(m)), interp(i), filter(std::move(filt)), homogeneous_boundaries(std::move(hom)),
          forward_stiffness(2 * mesh.num_nodes()),
          forward_stiffness_eigen(2 * mesh.num_nodes(), 2 * mesh.num_nodes()),
          nodal_forcing(std::move(forcing)), displacement(Eigen::VectorXd::Zero(nodal_forcing.size()))
    {
        assert(static_cast<unsigned long>(displacement.size()) == 2 * mesh.num_nodes());
    }

    ModelInfo(ModelInfo &&other) = default;
    // ModelInfo(const ModelInfo &other) = default;
};

template <int order>
using ModelInfoType = ModelInfo<MeshType<order>>;

using ModelInfoVariant =
    std::variant<ModelInfoType<1>, ModelInfoType<2>, ModelInfoType<3>, ModelInfoType<4>>;

// This is the main function you want to use from here.
std::unique_ptr<ModelInfoVariant> construct_model_info(
    const std::string &mesh_file, int order, Eigen::Vector2d force, ErsatzStiffness interp,
    double filt_radius, double lambda, double mu);

struct DirectDensitySpec {};

// And this one.
void update_model_info(ModelInfoVariant &minfo, const double *rho);
void update_model_info(ModelInfoVariant &minfo, const double *rho, DirectDensitySpec);

std::size_t ndofs(const ModelInfoVariant &minfo);
std::size_t num_elements(const ModelInfoVariant &minfo);

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
    construct_eigen_stiffness<Mesh>(minfo.forward_stiffness_eigen, minfo.forward_stiffness);
    minfo.factorized.analyzePattern(minfo.forward_stiffness_eigen);
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
    construct_eigen_stiffness<Mesh>(minfo.forward_stiffness_eigen, minfo.forward_stiffness);
    minfo.factorized.factorize(minfo.forward_stiffness_eigen);
}

template <class Mesh, class Vec>
void update_model_info(ModelInfo<Mesh> &minfo, const Vec &rho, DirectDensitySpec)
{
    assert(rho.size() == minfo.rho_filt.size());
    for (long i = 0; i < rho.size(); ++i)
    {
        minfo.rho_filt[i] = rho[i];
    }

    minfo.forward_stiffness.reset();
    for (std::size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        update_forward_stiffness(minfo, eli);
    }
    for (std::size_t which : minfo.homogeneous_boundaries)
    {
        Elasticity::TwoD::impose_homogeneous_condition(minfo.mesh, minfo.forward_stiffness, which);
    }
    construct_eigen_stiffness<Mesh>(minfo.forward_stiffness_eigen, minfo.forward_stiffness);
    minfo.factorized.factorize(minfo.forward_stiffness_eigen);
}

template <class Mesh>
void update_forward_stiffness(ModelInfo<Mesh> &minfo, std::size_t eli, bool init)
{
    const auto &nn = minfo.mesh.element(eli).node_numbers();
    const auto &Kel = minfo.ref_stiffness_matrices.at(eli);
    assert(nn.size() * 2 == static_cast<unsigned>(Kel.rows()));
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

template <class Mesh>
std::vector<std::size_t> get_dirichlet_boundaries(const Mesh &mesh)
{
    std::vector<std::size_t> ebounds;
    for (std::size_t i = 0; i < mesh.num_boundaries(); ++i)
    {
        if (mesh.boundary_tags(i).size() == 1 && mesh.boundary_tags(i)[0] == "ESSENTIAL")
        {
            fmt::print("Boundary {} is an essential boundary\n", i);
            ebounds.push_back(i);
        }
    }
    return ebounds;
}

template <class Mesh>
std::size_t get_forcing_boundary(const Mesh &mesh)
{
    for (std::size_t i = 0; i < mesh.num_boundaries(); ++i)
    {
        if (mesh.boundary_tags(i).size() == 1 && mesh.boundary_tags(i)[0] == "FORCE")
        {
            return i;
        }
    }
    return -1;
}

template <int Order>
auto build_vector_mass_matrix(const LineMesh<Order> &mesh)
{
    auto [elements, dofs] = mesh.get_elements_and_dofs();
    constexpr auto mass_form = [](const auto &u, const auto &v) { return u * v; };
    const auto num_dofs = dofs.back().back() + 1;

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    for (std::size_t el = 0; el < elements.size(); ++el)
    {
        const auto element = elements[el];
        const auto el_dofs = dofs[el];

        const auto local_M = element.form_matrix(mass_form);
        for (std::size_t i = 0; i < el_dofs.size(); ++i)
        {
            for (std::size_t j = i; j < el_dofs.size(); ++j)
            {
                triplets.emplace_back(2 * el_dofs[i], 2 * el_dofs[j], local_M(i, j));
                triplets.emplace_back(2 * el_dofs[i] + 1, 2 * el_dofs[j] + 1, local_M(i, j));
                if (j != i)
                {
                    triplets.emplace_back(2 * el_dofs[j], 2 * el_dofs[i], local_M(i, j));
                    triplets.emplace_back(2 * el_dofs[j] + 1, 2 * el_dofs[i] + 1, local_M(i, j));
                }
            }
        }
    }
    Eigen::SparseMatrix<double> M(num_dofs * 2, num_dofs * 2);
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

template <class Mesh>
void compute_pressure_force(
    Eigen::VectorXd &nodal_forcing, const Mesh &mesh, std::size_t fbound, Eigen::Vector2d f)
{
    Eigen::SparseMatrix<double> M = build_vector_mass_matrix(build_boundary_mesh(mesh, fbound));
    const auto &boundary = mesh.boundary(fbound);
    Eigen::VectorXd work(M.rows());
    for (int i = 0; i < M.rows(); i += 2)
    {
        work.segment(i, 2) = f;
    }

    work = (M * work).eval();

    for (std::size_t i = 0; i < boundary.nodes.size(); ++i)
    {
        std::size_t node = boundary.nodes[i];
        nodal_forcing.segment(node * 2, 2) += work.segment(i * 2, 2);
    }
}

#endif // FORWARD_MODEL_HPP
