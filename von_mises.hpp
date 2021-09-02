#ifndef VON_MISES_HPP
#define VON_MISES_HPP

#include <cassert>
#include <iostream>
#include <utility>

#include "2d.hpp"

namespace Elasticity
{

namespace TwoD
{

template <class Element>
struct VonMisesComputer
{
    template <class Vector>
    VonMisesComputer(const Element &el, const Vector &v, double lambda, double mu)
        : el(el), coeffs(), lambda{lambda}, mu{mu}
    {
        coeffs = v;
    }

    template <class... Args>
    double evaluate(const Args &...args) const
    {
        const auto dudx = partials(args...);
        const auto div = dudx[0] + dudx[1];
        const auto a = lambda * div + 2 * mu * dudx[0];
        const auto b = lambda * div + 2 * mu * dudx[1];
        const auto c = mu * (dudx[2] + dudx[3]);

        return std::sqrt(a * a - a * b + b * b + 3 * c * c);
    }

    template <class... Args>
    std::pair<double, Eigen::Matrix<double, 2 * basis_size<Element>, 1>>
    evaluate_with_gradient(const Args &...args) const
    {
        const auto dudx = partials(args...);
        const auto div = dudx[0] + dudx[1];
        const auto a = lambda * div + 2 * mu * dudx[0];
        const auto b = lambda * div + 2 * mu * dudx[1];
        const auto c = mu * (dudx[2] + dudx[3]);
        const auto sigma = std::sqrt(a * a - a * b + b * b + 3 * c * c);

        const double xcoeff1 = (lambda + 2 * mu) * (2 * a - b) + lambda * (2 * b - a);
        const double ycoeff1 = (lambda + 2 * mu) * (2 * b - a) + lambda * (2 * a - b);
        const double coeff2 = 6 * c * mu;
        Eigen::Matrix<double, 2 * basis_size<Element>, 1> grad;
        Galerkin::static_for<0, basis_size<Element>, 1>([&](auto I) {
            const double phi_x = el.template partial<0>(Galerkin::get<I()>(Element::basis))(args...);
            const double phi_y = el.template partial<1>(Galerkin::get<I()>(Element::basis))(args...);
            grad[2 * I()] = xcoeff1 * phi_x + coeff2 * phi_y;
            grad[2 * I() + 1] = ycoeff1 * phi_y + coeff2 * phi_x;
        });
        grad /= (2 * sigma);
        return std::make_pair(sigma, grad);
    }

    template <class Vector>
    void set_coeffs(const Vector &cs)
    {
        coeffs = cs;
    }

  private:
    const Element el;
    Eigen::Matrix<double, 2 * Element::basis.size(), 1> coeffs;
    double lambda, mu;

    // returns (ux_x, uy_y, ux_y, uy_x)
    template <class... Args>
    Eigen::Vector4d partials(const Args &...args) const
    {
        Eigen::Vector4d ps(0, 0, 0, 0);
        Galerkin::static_for<0, basis_size<Element>, 1>([&](const auto I) {
            const auto phi_x = el.template partial<0>(Galerkin::get<I()>(Element::basis))(args...);
            const auto phi_y = el.template partial<1>(Galerkin::get<I()>(Element::basis))(args...);
            ps[0] += coeffs[2 * I()] * phi_x;
            ps[1] += coeffs[2 * I() + 1] * phi_y;
            ps[2] += coeffs[2 * I()] * phi_y;
            ps[3] += coeffs[2 * I() + 1] * phi_x;
        });
        return ps;
    }
};

} // namespace TwoD

} // namespace Elasticity

#endif // VON_MISES_HPP
