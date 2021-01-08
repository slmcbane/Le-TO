/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef BILINEARQUADTRANSFORM_HPP
#define BILINEARQUADTRANSFORM_HPP

/*!
 * @file BilinearQuadTransform.hpp
 * @brief Implementation of a bilinear transformation of reference square.
 */

#include "TransformBase.hpp"
#include "Rationals.hpp"
#include "Polynomials.hpp"
#include "Quadrature.hpp"

#include <array>
#include <tuple>

namespace Galerkin
{

namespace Transforms
{

namespace
{

template <class T>
constexpr int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <class T1, class T2, class T3, class T4>
constexpr bool check_quad_geometry(const T1 &p1, const T2 &p2, const T3 &p3,
                                   const T4 &p4) noexcept
{
    constexpr auto triangle_det = [](const auto &p1, const auto &p2, const auto &p3)
    {
        const auto x1 = get<0>(p1);
        const auto x2 = get<0>(p2);
        const auto x3 = get<0>(p3);
        const auto y1 = get<1>(p1);
        const auto y2 = get<1>(p2);
        const auto y3 = get<1>(p3);

        return (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2);
    };

    const std::array<int, 4> dets {
        sgn(triangle_det(p1, p2, p3)),
        sgn(triangle_det(p2, p3, p4)),
        sgn(triangle_det(p3, p4, p1)),
        sgn(triangle_det(p4, p1, p2))
    };

    return dets[0] != 0 && dets[1] == dets[0] && dets[2] == dets[0] && dets[3] == dets[0];
}

} // namespace

/*!
 * @brief Bilinear mapping from reference square to quadrilaterals.
 *
 * Given four points defining the vertices of a quadrilateral, construct a
 * mapping from the reference square `[-1, 1] x [-1, 1]` to the quadrilateral
 * with the associated required functionality for `TransformBase`. There is an
 * option to construct the transform with a geometry check that ensures the
 * defined quadrilateral is valid, i.e. sides do not intersect and there are not
 * degenerate vertices.
 *
 * @see Galerkin::Transforms::TransformBase
 */
template <class T>
class BilinearQuadTransform : public TransformBase<2, BilinearQuadTransform<T>>
{
public:
    /*!
     * @brief Construct the transform with no geometry check from four vertices
     *
     * The arguments are vertices in the order in which they are connected. May
     * be clockwise or counter-clockwise; negative determinant of the transform
     * is OK.
     * 
     * The types of the points must support `get` to access
     * elements, via ADL. For example, `std::tuple` and `std::array` work out of
     * the box but not `std::vector`. You might define a type that works like so:
     * 
     *     struct Point
     *     {
     *         double x;
     *         double y;
     *     };
     * 
     *     template <auto I>
     *     constexpr double get(Point pt) noexcept
     *     {
     *         static_assert(I == 0 || I == 1);
     *         return I == 0 ? pt.x : pt.y;
     *     }
     */
    template <class T1, class T2, class T3, class T4>
    constexpr BilinearQuadTransform(const T1 &p1, const T2 &p2, const T3 &p3,
                                    const T4 &p4) noexcept : m_coeffs{0}
    {
        compute_coefficients(p1, p2, p3, p4);
    }

    /*!
     * @brief Construct the transform from points, with geometry check.
     */
    template <class T1, class T2, class T3, class T4>
    constexpr BilinearQuadTransform(const T1 &p1, const T2 &p2, const T3 &p3,
                                    const T4 &p4, GeometryCheck) : m_coeffs {0}
    {
        if (!check_quad_geometry(p1, p2, p3, p4))
        {
            throw GeometryException{};
        }
        compute_coefficients(p1, p2, p3, p4);
    }

    constexpr bool operator==(const BilinearQuadTransform<T> &other) const noexcept
    {
        return m_coeffs == other.m_coeffs;
    }

    template <class Arg>
    constexpr auto operator()(const Arg &pt) const noexcept
    {
        const auto xi = get<0>(pt);
        const auto eta = get<1>(pt);
        const auto x = m_coeffs[0] * xi * eta + m_coeffs[1] * xi + m_coeffs[2] * eta +
                       m_coeffs[3];
        const auto y = m_coeffs[4] * xi * eta + m_coeffs[5] * xi + m_coeffs[6] * eta +
                       m_coeffs[7];
        
        return std::array<std::remove_cv_t<decltype(x)>, 2>{x, y};
    }

    constexpr auto detJ() const noexcept
    {
        return Polynomials::Polynomial<T, Metanomials::Powers<1, 0>,
                                          Metanomials::Powers<0, 1>,
                                          Metanomials::Powers<0, 0>
                                      >(m_coeffs[1]*m_coeffs[4] - m_coeffs[0]*m_coeffs[5],
                                        m_coeffs[0]*m_coeffs[6] - m_coeffs[2]*m_coeffs[4],
                                        m_coeffs[1]*m_coeffs[6] - m_coeffs[2]*m_coeffs[5]);
    }

    template <int I, int J>
    constexpr auto jacobian() const noexcept
    {
        static_assert(I <= 1 && J <= 1 && I >= 0 && J >= 0);
        if constexpr (I == 0)
        {
            if constexpr (J == 0)
            {
                return Polynomials::Polynomial<T, Metanomials::Powers<0, 1>,
                                                  Metanomials::Powers<0, 0>
                                              >(m_coeffs[0], m_coeffs[1]);
            }
            else
            {
                return Polynomials::Polynomial<T, Metanomials::Powers<1, 0>,
                                                  Metanomials::Powers<0, 0>
                                              >(m_coeffs[0], m_coeffs[2]);
            }
        }
        else
        {
            if constexpr (J == 0)
            {
                return Polynomials::Polynomial<T, Metanomials::Powers<0, 1>,
                                                  Metanomials::Powers<0, 0>
                                              >(m_coeffs[4], m_coeffs[5]);
            }
            else
            {
                return Polynomials::Polynomial<T, Metanomials::Powers<1, 0>,
                                                  Metanomials::Powers<0, 0>
                                              >(m_coeffs[4], m_coeffs[6]);
            }
        }
    }

    template <int I, int J>
    constexpr auto inv_jacobian() const noexcept
    {
        if constexpr (I == 0)
        {
            if constexpr (J == 0)
            {
                return jacobian<1, 1>() / detJ();
            }
            else
            {
                return -jacobian<0, 1>() / detJ();
            }
        }
        else
        {
            if constexpr (J == 0)
            {
                return -jacobian<1, 0>() / detJ();
            }
            else
            {
                return jacobian<0, 0>() / detJ();
            }
        }
    }

    template <int I, class F>
    constexpr auto quadrature(F &&f) const noexcept
    {
        constexpr auto npoints = (I + 1) / 2 + (I - 1) % 2;
        if constexpr (npoints <= 0)
        {
            return Quadrature::box_integrate<2>(std::forward<F>(f), Quadrature::legendre_rule<T, 1>);
        }
        else
        {
            return Quadrature::box_integrate<2>(std::forward<F>(f), Quadrature::legendre_rule<T, npoints>);
        }
    }

private:
    std::array<T, 8> m_coeffs;

    template <class... Ps>
    constexpr void compute_coefficients(const Ps&... ps) noexcept
    {
        static_assert(sizeof...(Ps) == 4);
        const auto xs = std::tuple(get<0>(ps)...);
        const auto ys = std::tuple(get<1>(ps)...);
        m_coeffs[0] = Rationals::rational<1, 4> * 
            (get<0>(xs) - get<1>(xs) + get<2>(xs) - get<3>(xs));
        m_coeffs[1] = Rationals::rational<1, 4> *
            (-get<0>(xs) - get<1>(xs) + get<2>(xs) + get<3>(xs));
        m_coeffs[2] = Rationals::rational<1, 4> *
            (-get<0>(xs) + get<1>(xs) + get<2>(xs) - get<3>(xs));
        m_coeffs[3] = Rationals::rational<1, 4> *
            (get<0>(xs) + get<1>(xs) + get<2>(xs) + get<3>(xs));
        m_coeffs[4] = Rationals::rational<1, 4> *
            (get<0>(ys) - get<1>(ys) + get<2>(ys) - get<3>(ys));
        m_coeffs[5] = Rationals::rational<1, 4> *
            (-get<0>(ys) - get<1>(ys) + get<2>(ys) + get<3>(ys));
        m_coeffs[6] = Rationals::rational<1, 4> *
            (-get<0>(ys) + get<1>(ys) + get<2>(ys) - get<3>(ys));
        m_coeffs[7] = Rationals::rational<1, 4> *
            (get<0>(ys) + get<1>(ys) + get<2>(ys) + get<3>(ys));
    }
};

/*!
 * @brief Construct a `BilinearQuadTransform<double>` from points
 */
template <class T1, class T2, class T3, class T4>
constexpr auto bilinear_quad(const T1 &p1, const T2 &p2, const T3 &p3, const T4 &p4) noexcept
{
    return BilinearQuadTransform<double>(p1, p2, p3, p4);
}

/*!
 * @brief Construct a `BilinearQuadTransform<double>` from points, w/ geometry check.
 */
template <class T1, class T2, class T3, class T4>
constexpr auto bilinear_quad(const T1 &p1, const T2 &p2, const T3 &p3, const T4 &p4, GeometryCheck)
{
    return BilinearQuadTransform<double>(p1, p2, p3, p4, GeometryCheck{});
}

/********************************************************************************
 * Test the bilinear quadrilateral transform.
 *******************************************************************************/
#ifdef DOCTEST_LIBRARY_INCLUDED

#include <cmath>

TEST_CASE("[Galerkin::Transforms] Test transform constructors")
{
    // Simple translation.
    constexpr auto transform1 = BilinearQuadTransform<double>(
        std::tuple(0.0, 0.0), std::tuple(0.0, 1.0), std::tuple(1.0, 1.0),
        std::tuple(1.0, 0.0));

    // Should construct exactly the same transform.
    constexpr BilinearQuadTransform<double> transform2 =
        bilinear_quad(std::tuple(0, 0), std::tuple(0, 1), std::tuple(1, 1),
                      std::tuple(1, 0));

    REQUIRE(transform1 == transform2);

    // Can construct with a different type for coefficients if desired.
    REQUIRE_NOTHROW([[maybe_unused]] constexpr BilinearQuadTransform<float> transform3(
        std::tuple(0, 0), std::tuple(0, 1), std::tuple(1, 1), std::tuple(1, 0)));

    // Should be able to construct a transform with numbering of vertices
    // reversed.
    REQUIRE_NOTHROW([[maybe_unused]] constexpr auto transform4 =
                        bilinear_quad(std::tuple(0, 0), std::tuple(1, 0), std::tuple(1, 1),
                                      std::tuple(0, 1),
                                      GeometryCheck{}));

    // A non-convex quadrilateral should throw an exception when constructed
    // with a GeometryCheck argument, however.
    REQUIRE_THROWS_AS([[maybe_unused]] BilinearQuadTransform<double>
                          transform5(std::tuple(0, 0), std::tuple(0, 1), std::tuple(0.25, 0.25),
                                     std::tuple(1.0, 0.0), GeometryCheck{}),
                      GeometryException);

    // It should not throw an exception when constructed without the
    // optional GeometryCheck
    REQUIRE_NOTHROW([[maybe_unused]] constexpr auto transform6 = bilinear_quad(
                        std::tuple(0, 0), std::tuple(0, 1), std::tuple(0.25, 0.25),
                        std::tuple(1.0, 0)));

    // It should throw when numbering of vertices is reversed, too.
    REQUIRE_THROWS_AS([[maybe_unused]] BilinearQuadTransform<double>
                          transform7(std::tuple(0.5, 0.5), std::tuple(0, 0), std::tuple(1.0, 0.5),
                                     std::tuple(0.0, 1.0), GeometryCheck{}),
                      GeometryException);
} // TEST_CASE

TEST_CASE("[Galerkin::Transforms] Test that points are mapped correctly under a bilinear transform")
{
    SUBCASE("A simple uniform scaling and translation")
    {
        constexpr auto transform = bilinear_quad(
            std::tuple(0, 0), std::tuple(0, 1), std::tuple(1, 1), std::tuple(1, 0)
        );

        auto transformed = transform(std::tuple(0, 0));
        REQUIRE(get<0>(transformed) == doctest::Approx(0.5));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.5));

        transformed = transform(std::tuple(-0.5, 1));
        REQUIRE(get<0>(transformed) == doctest::Approx(0.25));
        REQUIRE(get<1>(transformed) == doctest::Approx(1.0));

        transformed = transform(std::array<double, 2>{0.75, -0.75});
        REQUIRE(get<0>(transformed) == doctest::Approx(0.875));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.125));

        transformed = transform(std::tuple(Rationals::rational<3, 4>, -Rationals::rational<3, 4>));
        REQUIRE(get<0>(transformed) == doctest::Approx(0.875));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.125));
    }

    SUBCASE("The same domain but with node numbering reversed")
    {
        constexpr auto transform = bilinear_quad(
            std::tuple(0, 0), std::tuple(1, 0), std::tuple(1, 1), std::tuple(0, 1)
        );

        auto transformed = transform(std::tuple(0, 0));
        REQUIRE(get<0>(transformed) == doctest::Approx(0.5));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.5));

        transformed = transform(std::tuple(-0.5, 1));
        REQUIRE(get<0>(transformed) == doctest::Approx(1.0));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.25));

        transformed = transform(std::tuple(Rationals::rational<3, 4>, -Rationals::rational<3, 4>));
        REQUIRE(get<0>(transformed) == doctest::Approx(0.125));
        REQUIRE(get<1>(transformed) == doctest::Approx(0.875));
    }

    SUBCASE("A more general transformation")
    {
        constexpr auto transform = bilinear_quad(
            std::tuple(0, 0), std::tuple(Rationals::rational<1, 4>, 1),
            std::tuple(2, Rationals::rational<7, 4>), std::tuple(2, 0)
        );

        auto transformed = transform(std::tuple(0, 0));
        REQUIRE(get<0>(transformed) == doctest::Approx(17.0 / 16));
        REQUIRE(get<1>(transformed) == doctest::Approx(11.0 / 16));

        transformed = transform(std::array<double, 2>{-0.5, 1.0});
        REQUIRE(get<0>(transformed) == doctest::Approx(11.0 / 16));
        REQUIRE(get<1>(transformed) == doctest::Approx(19.0 / 16));
    }
}

TEST_CASE("[Galerkin::Transforms] Test Jacobian computations with bilinear transform")
{
    constexpr auto transform = bilinear_quad(
        std::tuple(0, 0), std::tuple(Rationals::rational<1, 4>, 1),
        std::tuple(2, Rationals::rational<7, 4>), std::tuple(2, 0)
    );

    constexpr auto detj = transform.detJ();

    REQUIRE(detj(std::tuple(0, 0)) == doctest::Approx(81.0 / 128));
    REQUIRE(detj(std::tuple(0.5, -0.5)) == doctest::Approx(193.0 / 256));

    REQUIRE(transform.jacobian<0, 0>()(std::tuple(0, 0)) == doctest::Approx(15.0 / 16));
    REQUIRE(transform.jacobian<0, 1>()(std::tuple(0, 0)) == doctest::Approx(1.0 / 16));
    REQUIRE(transform.jacobian<1, 0>()(std::tuple(0, 0)) == doctest::Approx(3.0 / 16));
    REQUIRE(transform.jacobian<1, 1>()(std::tuple(0, 0)) == doctest::Approx(11.0 / 16));

    REQUIRE(transform.inv_jacobian<0, 0>()(std::tuple(0, 0)) == doctest::Approx(88.0 / 81));
    REQUIRE(transform.inv_jacobian<0, 1>()(std::tuple(0, 0)) == doctest::Approx(-8.0 / 81));
    REQUIRE(transform.inv_jacobian<1, 0>()(std::tuple(0, 0)) == doctest::Approx(-8.0 / 27));
    REQUIRE(transform.inv_jacobian<1, 1>()(std::tuple(0, 0)) == doctest::Approx(40.0 / 27));
} // TEST_CASE

TEST_CASE("[Galerkin::Transforms] Exercise partials and integration for bilinear transform")
{
    constexpr auto transform = bilinear_quad(
        std::tuple(0, 0), std::tuple(1, 1), std::tuple(3, 1), std::tuple(2, 0),
        GeometryCheck{}
    );

    constexpr auto f = Polynomials::Polynomial<double, Metanomials::Powers<1, 1>,
                                                       Metanomials::Powers<1, 0>,
                                                       Metanomials::Powers<0, 1>,
                                                       Metanomials::Powers<0, 0>
                                              >(0.25, -0.25, -0.25, 0.25);
    
    REQUIRE(transform.integrate<2>(f) == doctest::Approx(0.5));
    
    constexpr auto dx = transform.partial<0>(f);
    constexpr auto dy = transform.partial<1>(f);

    REQUIRE(transform.integrate<2>(Functions::ConstantFunction(1)) == doctest::Approx(2.0));
    REQUIRE(transform.integrate<2>(Metanomials::metanomial(
        Metanomials::term(Rationals::rational<1>, Metanomials::Powers<2, 0>())
    )) == doctest::Approx(2.0 / 3));
    REQUIRE(transform.integrate<2>(Metanomials::metanomial(
        Metanomials::term(Rationals::rational<1>, Metanomials::Powers<2, 2>())
    )) == doctest::Approx(2.0 / 9));

    REQUIRE(transform.integrate<2>(dx*dx + dy*dy) == doctest::Approx(0.5));
} // TEST_CASE

#endif

/********************************************************************************
 * End test block.
 *******************************************************************************/

} // namespace Transforms

} // namespace Galerkin

#endif /* BILINEARQUAD_HPP */
