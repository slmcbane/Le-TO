/*
 * Copyright (c) 2019, Sean McBane
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

/*!
 * @file Quadrature.hpp
 * @brief Implementation of quadrature rules for numerical integration
 */

#include <array>
#include "utils.hpp"
#include "Legendre.hpp"

namespace Galerkin
{

/// All functionality for numerical integration is in this namespace
namespace Quadrature
{

/*!
 * @brief Representation of a quadrature rule
 * 
 * A quadrature rule consists of a list of points and weights; the integral of
 * a function `f` is `sum(f(points[i]) * weights[i])`.
 */
template <class Point, class Weight, auto N>
struct Rule
{
    const std::array<Point, N> points;
    const std::array<Weight, N> weights;
};

/*!
 * @brief Integrate function `f` using the given quadrature rule.
 */
template <class F, class P, class W, auto N>
constexpr auto integrate(const F &f, const Rule<P, W, N> &rule) noexcept
{
    auto x = zero<decltype(f(rule.points[0]) * rule.weights[0])>;
    for (int i = 0; i < N; ++i)
    {
        x += f(rule.points[i]) * rule.weights[i];
    }
    return x;
}

/*!
 * @brief Integrate function `f` over `Dim` dimensional box
 * 
 * Given a function `f: R^n -> R`, integrate `f` over the box `[-1, 1]^n`. This
 * is done by simply nesting one dimensional quadrature rules.
 */
template <auto Dim, class F, class P, class W, auto N>
constexpr auto box_integrate(const F &f, const Rule<P, W, N> &rule) noexcept
{
    static_assert(Dim >= 1);
    if constexpr (Dim == 1)
    {
        return integrate(
            [&](auto x) { return f(std::tuple(x)); },
            rule
        );
    }
    else
    {
        constexpr auto bind_head = [](const auto &g, auto x)
        {
            return [=, &g](auto tail) { return g(std::tuple_cat(std::tuple(x), tail)); };
        };
        return integrate(
            [&](auto x) { return box_integrate<Dim - 1>(bind_head(f, x), rule); },
            rule
        );
    }
}

/// Compute the weights for a Gauss-Legendre rule.
template <class T, auto N>
constexpr auto legendre_weights() noexcept
{
    std::array<T, N> weights{};
    constexpr std::array<T, N> roots = Legendre::roots<T, N>;
    constexpr auto pprime = partial<0>(Legendre::polynomial<N>);
    int i = 0;
    for (T x : roots)
    {
        weights[i++] = 2 / ((1 - x * x) * pprime(x) * pprime(x));
    }
    return weights;
}

/********************************************************************************
 * Test that computed quadrature points are correct.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

#include <cmath>

TEST_CASE("Test compute points for Gauss-Legendre quadrature")
{
    REQUIRE(legendre_weights<double, 1>()[0] == doctest::Approx(2.0));
    REQUIRE(legendre_weights<double, 2>()[0] == doctest::Approx(1.0));
    REQUIRE(legendre_weights<double, 2>()[1] == doctest::Approx(1.0));
    REQUIRE(legendre_weights<double, 3>()[0] == doctest::Approx(5.0 / 9));
    REQUIRE(legendre_weights<double, 3>()[1] == doctest::Approx(8.0 / 9));
    REQUIRE(legendre_weights<double, 3>()[2] == doctest::Approx(5.0 / 9));
    REQUIRE(legendre_weights<double, 4>()[0] == doctest::Approx((18 - std::sqrt(30)) / 36));
    REQUIRE(legendre_weights<double, 4>()[1] == doctest::Approx((18 + std::sqrt(30)) / 36));
    REQUIRE(legendre_weights<double, 4>()[2] == doctest::Approx((18 + std::sqrt(30)) / 36));
    REQUIRE(legendre_weights<double, 4>()[3] == doctest::Approx((18 - std::sqrt(30)) / 36));
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 * End test of computed quadrature points.
 *******************************************************************************/

/*!
 * @brief Gauss-Legendre quadrature rule with `N` points.
 * 
 * This quadrature rule over the interval `(-1, 1)` is exact for polynomials of
 * degree <= `2N-1`.
 */
template <class T, auto N>
constexpr Rule<T, T, N> legendre_rule = Rule<T, T, N> { Legendre::roots<T, N>,
                                                        legendre_weights<T, N>(),
                                                      };

/********************************************************************************
 * Test that an integral is computed accurately.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("Test that integrals are computed exactly")
{
    SUBCASE("Check the integral of an order 1 polynomial with 1 quadrature point")
    {
        REQUIRE(
            integrate( [](double x) { return 3*x + 2; }, legendre_rule<double, 1> )
                == doctest::Approx(3.0 / 2 + 2 - (3.0 / 2 - 2))
        );
    }

    SUBCASE("Check the integral of order 2 and order 3 polynomials with 2 points")
    {
        REQUIRE(
            integrate( [](double x) { return 3*x*x + 2*x + 1; },
                       legendre_rule<double, 2> ) ==
            doctest::Approx(4.0)
        );

        REQUIRE(
            integrate( [](double x) { return 4*x*x*x + 3*x*x + 2*x + 1; },
                       legendre_rule<double, 2> ) ==
            doctest::Approx(4.0)
        );
    }

    SUBCASE("Check the integral of an order 4 polynomial with 3 points")
    {
        constexpr auto f = [](double x)
        {
            return x*x*x*x + x*x*x + x*x + x + 1;
        };

        constexpr auto F = [](double x)
        {
            return x*x*x*x*x / 5 + x*x*x*x / 4 + x*x*x / 3 + x*x / 2 + x;
        };

        REQUIRE(integrate(f, legendre_rule<double, 3>) == doctest::Approx(F(1.0) - F(-1.0)));
    }
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 * End test block
 *******************************************************************************/

/*!
 * @brief Compute a rule with N^2 points for quadrature on a triangle.
 * 
 * @see triangle_rule
 */
template <class T, auto N>
constexpr auto compute_triangle_rule() noexcept
{
    std::array<T, N*N> weights {};
    std::array<std::tuple<T, T>, N*N> points {};
    constexpr auto rule = legendre_rule<T, N>;
    for (int i = 0; i < N; ++i)
    {
        const T eta = rule.points[i];
        for (int j = 0; j < N; ++j)
        {
            const T xi = rule.points[j];
            weights[i*N+j] = rule.weights[i] * rule.weights[j] * (2 - xi - eta) / 4;
            get<0>(points[i*N+j]) = (-xi*eta + 3*xi - eta - 1) / 4;
            get<1>(points[i*N+j]) = (-xi*eta - xi + 3*eta - 1) / 4;
        }
    }
    return Rule<std::tuple<T, T>, T, N*N>{points, weights};
}

/*!
 * @brief Quadrature rule for reference triangle
 * 
 * This rule is derived by mapping the reference square `[-1, 1] x [-1, 1]` to
 * a reference triangle defined by the vertices `(-1, -1), (-1, 1), (1, -1)`.
 * This isn't the optimal way to construct a quadrature rule, but it was easy to
 * implement using existing functionality. The rule given by
 * `triangle_rule<T, N>` is exact for polynomials of total degree <= `2N - 2`;
 * for the box quadrature only the degree of the polynomial matters and the
 * degree must be <= `2N - 1`.
 */
template <class T, auto N>
constexpr Rule<std::tuple<T, T>, T, N*N> triangle_rule = compute_triangle_rule<T, N>();

/********************************************************************************
 * Test integration using a triangle rule.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Quadrature] Test integration using triangular rule")
{

SUBCASE("Test integration of order 0 polynomial with 1 point and 2 points")
{
    constexpr auto f = []([[maybe_unused]] auto t) { return 1.0; };

    REQUIRE(integrate(f, triangle_rule<double, 1>) == doctest::Approx(2.0));
    REQUIRE(integrate(f, triangle_rule<double, 2>) == doctest::Approx(2.0));
} // SUBCASE

SUBCASE("Test integration of order 1 polynomial with 2 points and 3 points")
{
    constexpr auto f = [](auto t) { return 2*get<0>(t) - get<1>(t); };

    REQUIRE(integrate(f, triangle_rule<double, 2>) == doctest::Approx(-2.0 / 3));
    REQUIRE(integrate(f, triangle_rule<double, 3>) == doctest::Approx(-2.0 / 3));
} // SUBCASE

SUBCASE("Test integration of order 2 polynomial with 2 points and 3 points")
{
    constexpr auto f = [](auto t) { return get<0>(t) * get<0>(t); };

    REQUIRE(integrate(f, triangle_rule<double, 2>) == doctest::Approx(2.0 / 3));
    REQUIRE(integrate(f, triangle_rule<double, 3>) == doctest::Approx(2.0 / 3));
} // SUBCASE

SUBCASE("Test integration of order 3 polynomial")
{
    constexpr auto f = [](auto t) { return get<0>(t)*get<0>(t)*get<0>(t) * 
                                           get<1>(t)*get<1>(t)*get<1>(t) -
                                           get<0>(t)*get<0>(t); };

    REQUIRE(integrate(f, triangle_rule<double, 4>) == doctest::Approx(-2.0 / 3));

    constexpr auto g = [](auto t) { return get<0>(t)*get<0>(t)*get<0>(t); };

    REQUIRE(integrate(g, triangle_rule<double, 3>) == doctest::Approx(-0.4));
} // SUBCASE

} // TEST_CASE

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End test block
 *******************************************************************************/

} /* namespace Quadrature */

} /* namespace Galerkin */

#endif /* QUADRATURE_HPP */