/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef TRIANGLETRANSFORM_HPP
#define TRIANGLETRANSFORM_HPP

/*!
 * @file TriangleTransform.hpp
 * @brief Implementation of transformation from reference to instantiated triangle domain.
 */

#include "FunctionBase.hpp"
#include "Quadrature.hpp"
#include "Rationals.hpp"
#include "TransformBase.hpp"

#include <array>
#include <type_traits>

namespace Galerkin
{

namespace Transforms
{

/*!
 * @brief Implements coordinate mapping from reference triangle.
 * 
 * A `TriangleTransform` represents the coordinate map from the reference
 * triangle defined by vertices (-1, -1), (-1, 1), (1, -1), in that order. It
 * implements the interface required to use functionality of `TransformBase`.
 */
template <class T>
class TriangleTransform : public TransformBase<2, TriangleTransform<T>>
{
public:
    /*!
     * @brief Construct the map from 3 vertices
     * 
     * The three vertices correspond to the vertices of the reference triangle
     * as listed in the class's general documentation. The types of the passed
     * vertices must implement element access through a `get` function found via
     * ADL - e.g. `std::tuple`, `std::array`, or a custom type as described in
     * the documentation for `BilinearQuadTransform`.
     */
    template <class P1, class P2, class P3>
    constexpr TriangleTransform(const P1 &p1, const P2 &p2, const P3 &p3) noexcept :
        m_coeffs{ 0 }
    {
        auto x1 = get<0>(p1); auto y1 = get<1>(p1);
        auto x2 = get<0>(p2); auto y2 = get<1>(p2);
        auto x3 = get<0>(p3); auto y3 = get<1>(p3);

        m_coeffs[0] = x3 - x1;
        m_coeffs[1] = x2 - x1;
        m_coeffs[2] = x2 + x3;
        m_coeffs[3] = y3 - y1;
        m_coeffs[4] = y2 - y1;
        m_coeffs[5] = y2 + y3;
        for (auto &coeff: m_coeffs)
        {
            coeff /= 2;
        }
    }

    template <class Arg>
    constexpr auto operator()(const Arg &arg) const noexcept
    {
        const auto xi = get<0>(arg);
        const auto eta = get<1>(arg);
        const auto x = m_coeffs[0] * xi + m_coeffs[1] * eta + m_coeffs[2];
        const auto y = m_coeffs[3] * xi + m_coeffs[4] * eta + m_coeffs[5];
        return std::array<std::remove_cv_t<decltype(x)>, 2>{ x, y };
    }

    constexpr auto detJ() const noexcept
    {
        return Functions::ConstantFunction(m_coeffs[0] * m_coeffs[4] -
                                           m_coeffs[1] * m_coeffs[3]);
    }

    template <int I, int J>
    constexpr auto jacobian() const noexcept
    {
        static_assert(I <= 1 && J <= 1 && I >= 0 && J >= 0);
        if constexpr (I == 0)
        {
            if constexpr (J == 0)
            {
                return Functions::ConstantFunction(m_coeffs[0]);
            }
            else
            { 
                return Functions::ConstantFunction(m_coeffs[1]);
            }
        }
        else
        {
            if constexpr (J == 0)
            { 
                return Functions::ConstantFunction(m_coeffs[3]);
            }
            else
            {
                return Functions::ConstantFunction(m_coeffs[4]);
            }
        }
    }

    template <int I, int J>
    constexpr auto inv_jacobian() const noexcept
    {
        static_assert(I <= 1 && J <= 1 && I >= 0 && J >= 0);
        const auto determinant = m_coeffs[0] * m_coeffs[4] - m_coeffs[1] * m_coeffs[3];
        if constexpr (I == 0)
        {
            if constexpr (J == 0)
            {
                return Functions::ConstantFunction(m_coeffs[4] / determinant);
            }
            else
            {
                return Functions::ConstantFunction(-m_coeffs[1] / determinant);
            }
        }
        else
        {
            if constexpr (J == 0)
            {
                return Functions::ConstantFunction(-m_coeffs[3] / determinant);
            }
            else
            {
                return Functions::ConstantFunction(m_coeffs[0] / determinant);
            }
        }
    }

    template <int I, class F>
    constexpr auto quadrature(F &&f) const noexcept
    {
        // To integrate a polynomial with total degree D, we need the number of
        // quadrature points N to satisfy 2N - 2 >= D. Adding D % 2 to the
        // computed N = (D+2) / 2 corrects for the case where D is odd.
        constexpr auto npoints = (I + 2) / 2 + I % 2 > 0 ? (I + 2) / 2 + I % 2 : 1;
        return Quadrature::integrate(std::forward<F>(f), Quadrature::triangle_rule<T, npoints>);
    }

private:
    std::array<T, 6> m_coeffs;
};

/*!
 * @brief Helper to construct a double precision triangle transform.
 * 
 * This function simply forwards the arguments to the `TriangleTransform<double>`
 * constructor for the most common case.
 */
template <class... Ps>
constexpr auto triangle_transform(const Ps &...ps) noexcept
{
    static_assert(sizeof...(Ps) == 3);
    return TriangleTransform<double>(ps...);
}

/********************************************************************************
 * Test block for triangle transform
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Transforms] Test mapping of a triangle")
{

constexpr auto transform = triangle_transform(
    std::tuple(0.5, 0.5), std::tuple(0.0, 1.0), std::tuple(1.0, 1.0)
);

SUBCASE("Check transformations of points")
{
    auto pt = transform(std::tuple(-1.0, -1.0));
    REQUIRE(get<0>(pt) == doctest::Approx(0.5));
    REQUIRE(get<1>(pt) == doctest::Approx(0.5));

    pt = transform(std::tuple(-0.5, -0.5));
    REQUIRE(get<0>(pt) == doctest::Approx(0.5));
    REQUIRE(get<1>(pt) == doctest::Approx(0.75));

    pt = transform(std::tuple(-0.5, 0.5));
    REQUIRE(get<0>(pt) == doctest::Approx(0.25));
    REQUIRE(get<1>(pt) == doctest::Approx(1.0));
} // SUBCASE

SUBCASE("Check the Jacobian elements and determinant")
{
    constexpr auto arg = std::tuple(0.0, -0.1);
    REQUIRE(transform.detJ()(arg) == doctest::Approx(1.0 / 8));
    REQUIRE(transform.jacobian<0, 0>()(arg) == doctest::Approx(0.25));
    REQUIRE(transform.jacobian<0, 1>()(arg) == doctest::Approx(-0.25));
    REQUIRE(transform.jacobian<1, 0>()(arg) == doctest::Approx(0.25));
    REQUIRE(transform.jacobian<1, 1>()(arg) == doctest::Approx(0.25));
} // SUBCASE

} // TEST_CASE

TEST_CASE("[Galerkin::Transforms] Test integration through a triangle transform")
{
    /*
     * This test case uses a transformation mapping to the triangle with vertices
     * (0, 0), (0, 1), (3/2, 1). I integrated by hand the polynomial x^2 * y over
     * this region, and the test checks that integrating the corresponding
     * polynomial in the reference domain returns the correct result, which is
     * 9 / 40.
     */
    constexpr auto transform = triangle_transform(std::tuple(0, 0),
                                                  std::tuple(0, 1),
                                                  std::tuple(1.5, 1));

    constexpr auto poly1 = Metanomials::metanomial(
        Metanomials::term(Rationals::rational<3, 4>, Metanomials::Powers<1, 0>{}),
        Metanomials::term(Rationals::rational<3, 4>, Metanomials::Powers<0, 0>{})
    );
    constexpr auto poly2 = Metanomials::metanomial(
        Metanomials::term(Rationals::rational<1, 2>, Metanomials::Powers<1, 0>{}),
        Metanomials::term(Rationals::rational<1, 2>, Metanomials::Powers<0, 1>{}),
        Metanomials::term(Rationals::rational<1>, Metanomials::Powers<0, 0>{})
    );
    constexpr auto integrand = poly1 * poly1 * poly2;

    REQUIRE(transform.integrate<3>(integrand) == doctest::Approx(9.0 / 40));
} // TEST_CASE

#endif // DOCTEST_LIBRARY_INCLUDED

} // namespace Transforms

} // namespace Galerkin

#endif // TRIANGLETRANSFORM_HPP
