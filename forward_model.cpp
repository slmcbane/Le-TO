#include "forward_model.hpp"
#include "ersatz_stiffness.hpp"

namespace
{

ModelInfoVariant construct_model_info_internal(
    const std::string &mesh_file, int order, Eigen::Vector2d force, ErsatzStiffness interp,
    double filt_radius)
{
    assert(order >= 1 && order <= 4);
    MeshVariant mesh = read_mesh(mesh_file.c_str(), order);
    std::vector<std::size_t> ebounds =
        std::visit([](const auto &mesh) { return get_dirichlet_boundaries(mesh); }, mesh);

    std::size_t fbound = std::visit([](const auto &mesh) { return get_forcing_boundary(mesh); }, mesh);
    Eigen::VectorXd nodal_forcing =
        Eigen::VectorXd::Zero(std::visit([](const auto &mesh) { return mesh.num_nodes() * 2; }, mesh));
    std::visit(
        [&, force](const auto &mesh) { compute_pressure_force(nodal_forcing, mesh, fbound, force); }, mesh);

    auto filter = build_filters(mesh, filt_radius);

    switch (order)
    {
    case 1:
        return ModelInfoVariant(
            std::in_place_type_t<ModelInfoType<1>>{}, std::move(std::get<0>(mesh)), interp,
            std::move(filter), std::move(ebounds), std::move(nodal_forcing));
    case 2:
        return ModelInfoVariant(
            std::in_place_type_t<ModelInfoType<2>>{}, std::move(std::get<1>(mesh)), interp,
            std::move(filter), std::move(ebounds), std::move(nodal_forcing));
    case 3:
        return ModelInfoVariant(
            std::in_place_type_t<ModelInfoType<3>>{}, std::move(std::get<2>(mesh)), interp,
            std::move(filter), std::move(ebounds), std::move(nodal_forcing));
    default:
        return ModelInfoVariant(
            std::in_place_type_t<ModelInfoType<4>>{}, std::move(std::get<3>(mesh)), interp,
            std::move(filter), std::move(ebounds), std::move(nodal_forcing));
    }
}

} // namespace

std::unique_ptr<ModelInfoVariant> construct_model_info(
    const std::string &mesh_file, int order, Eigen::Vector2d force, ErsatzStiffness interp,
    double filt_radius, double lambda, double mu)
{
    std::unique_ptr<ModelInfoVariant> ptr(
        new ModelInfoVariant(construct_model_info_internal(mesh_file, order, force, interp, filt_radius)));
    std::visit([=](auto &model_info) { initialize_model_info(model_info, lambda, mu); }, *ptr);
    return ptr;
}

std::size_t ndofs(const ModelInfoVariant &minfo)
{
    return std::visit([](const auto &minfo) { return 2 * minfo.mesh.num_nodes(); }, minfo);
}

std::size_t num_elements(const ModelInfoVariant &minfo)
{
    return std::visit([](const auto &minfo) { return minfo.mesh.num_elements(); }, minfo);
}

void update_model_info(ModelInfoVariant &minfo, const double *rho)
{
    std::visit(
        [rho](auto &m) {
            update_model_info(m, Eigen::Map<const Eigen::VectorXd>(rho, m.mesh.num_elements()));
        },
        minfo);
}

void update_model_info(ModelInfoVariant &minfo, const double *rho, DirectDensitySpec)
{
    std::visit(
        [rho](auto &m) {
            update_model_info(
                m, Eigen::Map<const Eigen::VectorXd>(rho, m.mesh.num_elements()), DirectDensitySpec{});
        },
        minfo);
}
