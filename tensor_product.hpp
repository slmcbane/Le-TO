#ifndef TENSOR_PRODUCT_HPP
#define TENSOR_PRODUCT_HPP

#include "forward_model.hpp"

#include <vector>

/*
 * Compute $\partial K/\partial rho times the displacement vector.
 * Assumes that the problem has already been solved for displacement.
 */
template <class U, class V>
void evaluate_tensor_product(U &grad, const V &v, const ModelInfo<MeshType<1>> &minfo)
{
    assert(grad.size() == minfo.mesh.num_elements());
    assert(v.size() == 2 * minfo.mesh.num_nodes());
    assert(minfo.rho_filt.size() == minfo.mesh.num_elements());

    grad.fill(0);
    const auto &refmats = minfo.ref_stiffness_matrices;
    Eigen::Matrix<double, ModelInfo<MeshType<1>>::el_stiffness_matrix_t::RowsAtCompileTime, 1> uel, vel;

    for (std::size_t i = 0; i < minfo.mesh.num_elements(); ++i)
    {
        double dsdrho = minfo.interp.derivative(minfo.rho_filt[i]);
        const auto &Kel = refmats[i];
        const auto &efilt = minfo.filter[i];

        const auto &nn = minfo.mesh.element(i).node_numbers();
        for (std::size_t j = 0; j < nn.size(); ++j)
        {
            uel[2 * j] = minfo.displacement[2 * nn[j]];
            vel[2 * j] = v[2 * nn[j]];
            uel[2 * j + 1] = minfo.displacement[2 * nn[j] + 1];
            vel[2 * j + 1] = v[2 * nn[j] + 1];
        }

        const double vTdKdrhou = dsdrho * vel.dot(Kel * uel);
        for (unsigned i = 0; i < efilt.entries.size(); ++i)
        {
            grad[efilt.entries[i].index] += efilt.entries[i].weight * vTdKdrhou;
        }
    }
}

#endif // TENSOR_PRODUCT_HPP
