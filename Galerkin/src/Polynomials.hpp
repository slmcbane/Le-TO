/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include "FunctionBase.hpp"
#include "Metanomials.hpp"
#include "utils.hpp"

#include <array>

namespace Galerkin
{

namespace Polynomials
{

template <class T, class... Powers>
class Polynomial : public Functions::FunctionBase<Polynomial<T, Powers...>>
{
public:
    template <class... Args>
    constexpr Polynomial(Args... args) noexcept : m_coeffs{static_cast<T>(args)...}
    {
        static_assert(sizeof...(Args) == sizeof...(Powers));
    }

    constexpr Polynomial() : m_coeffs {} {}

    constexpr const T& operator[](int i) const noexcept { return m_coeffs[i]; }
    constexpr T& operator[](int i) noexcept { return m_coeffs[i]; }

    template <class X>
    constexpr auto operator()(const X &args) const noexcept
    {
        return static_sum<0, sizeof...(Powers)>(
            [&](auto I) {
                return m_coeffs[I()] * Metanomials::raise(args, get<I()>(typeconst_list<Powers...>()));
            },
            zero<T>
        );
    }

    constexpr auto& coeffs() const noexcept { return m_coeffs; }

    template <auto I>
    constexpr auto partial() const noexcept
    {
        if constexpr (sizeof...(Powers) == 0)
        {
            return Polynomial<T>();
        }
        else
        {
            static_assert(I < Metanomials::nvars(get<0>(typeconst_list<Powers...>())),
                          "Index for partial derivative out of bounds");
            constexpr auto inds = select_partial_terms<I>();
            constexpr auto N = std::tuple_size_v<decltype(inds)>;
            std::array<T, N> coeffs {};
            constexpr auto power_list = typeconst_list<Powers...>();

            auto new_powers = static_reduce<0, N, 1>(
                [&](auto J)
                {
                    constexpr auto index = get<J()>(inds);
                    auto power = get<index>(power_list);
                    coeffs[J()] = m_coeffs[index] * Metanomials::get_power<I>(power);
                    return Metanomials::subtract_one<I>(power);
                },
                typeconst_list<>(),
                [](auto l, auto p) { return l.append(make_list(p)); }
            );

            return construct_polynomial(coeffs, new_powers);
        }
    }

private:
    std::array<T, sizeof...(Powers)> m_coeffs;

    template <class Ty, class... Ps>
    static constexpr auto construct_polynomial(const std::array<Ty, sizeof...(Ps)> coeffs,
                                               typeconst_list<Ps...>) noexcept
    {
        auto p = Polynomial<Ty, Ps...>();
        for (unsigned i = 0; i < sizeof...(Ps); ++i)
        {
            p[i] = coeffs[i];
        }
        return p;
    }

    template <int v>
    struct PowerCombiner
    {
        template <class Tup>
        constexpr auto operator()(Tup tup) const noexcept
        {
            if constexpr (v != -1)
            {
                return std::tuple_cat(tup, std::tuple(v));
            }
            else
            {
                return tup;
            }
        }
    };

    template <int v>
    static constexpr auto power_combiner(std::integral_constant<int, v>) noexcept
    {
        return PowerCombiner<v>();
    }

    template <auto I>
    static constexpr auto select_partial_terms() noexcept
    {
        constexpr auto power_list = typeconst_list<Powers...>();
        return static_reduce<0, sizeof...(Powers), 1>(
            [=](auto J)
            {
                constexpr auto power = get<J()>(power_list);
                if constexpr (Metanomials::get_power<I>(power) != 0)
                {
                    return J;
                }
                else
                {
                    return intgr_constant<-1>;
                }
            },
            std::tuple<>(),
            [](auto tup, auto power)
            {
                return power_combiner(power)(tup);
            }
        );
    }
};

template <class T, class... Powers>
constexpr bool operator==(const Polynomial<T, Powers...> &p1,
                          const Polynomial<T, Powers...> &p2) noexcept
{
    return p1.coeffs() == p2.coeffs();
}

template <class T, class... P1s, class... P2s>
constexpr bool operator==(const Polynomial<T, P1s...>&, const Polynomial<T, P2s...>&) noexcept
{
    return false;
}

/********************************************************************************
 * Test polynomial construction, arithmetic operations, evaluation.
 *******************************************************************************/
#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Polynomials] Basic arithmetic and evaluation of polynomials")
{
    SUBCASE("A single-variable polynomial test case")
    {
        // p(x) = x^2 - 1
        constexpr auto p = Polynomial<int, Metanomials::Powers<2>, Metanomials::Powers<0>>(1, -1);

        // g(x) = x + 1;
        constexpr auto g = Polynomial<int, Metanomials::Powers<1>, Metanomials::Powers<0>>(1, 1);

        REQUIRE(p(std::tuple(1)) == 0);
        REQUIRE(p(std::tuple(1.5)) == doctest::Approx(1.25));
        REQUIRE(p(std::tuple(2)) == 3);

        REQUIRE(g(std::tuple(1)) == 2);
        REQUIRE(g(std::tuple(1.5)) == doctest::Approx(2.5));
        REQUIRE(g(std::tuple(2)) == 3);

        // Should work with an array argument, too.
        constexpr std::array x{3.0};
        REQUIRE(g(x) == doctest::Approx(4.0));

        // Check sum, product, and quotient of polynomials.
        constexpr auto p_plus_g = p + g;
        REQUIRE(p_plus_g(std::tuple(1)) == 2);
        REQUIRE(p_plus_g(std::tuple(1.5)) == doctest::Approx(3.75));
        REQUIRE(p_plus_g(std::tuple(2)) == 6);
        REQUIRE(p_plus_g(x) == doctest::Approx(12.0));

        constexpr auto p_times_g = p * g;
        REQUIRE(p_times_g(std::tuple(1)) == 0);
        REQUIRE(p_times_g(std::tuple(1.5)) == doctest::Approx(1.25 * 2.5));
        REQUIRE(p_times_g(std::tuple(2)) == 9);
        REQUIRE(p_times_g(x) == doctest::Approx(32.0));

        constexpr auto p_by_g = p / g;
        REQUIRE(p_by_g(std::tuple(1)) == 0);
        REQUIRE(p_by_g(std::tuple(1.5)) == doctest::Approx(1.25 / 2.5));
        REQUIRE(p_by_g(std::tuple(2)) == 1);
        REQUIRE(p_by_g(x) == doctest::Approx(2.0));
    }

    SUBCASE("Test functionality with a multi-variable polynomial")
    {
        // p(x, y) = x^2 + 2xy - 2y + 1
        constexpr auto p = Polynomial<double, Metanomials::Powers<2, 0>,
                                      Metanomials::Powers<1, 1>,
                                      Metanomials::Powers<0, 1>,
                                      Metanomials::Powers<0, 0>
                                     >(1, 2, -2, 1);
        // g(x, y) = y^2 - x + 2
        constexpr auto g = Polynomial<double, Metanomials::Powers<0, 2>,
                                      Metanomials::Powers<1, 0>, Metanomials::Powers<0, 0>
                                     >(1, -1, 2);

        // Evaluation points
        constexpr auto point1 = std::tuple(Rationals::rational<1, 2>, -Rationals::rational<1, 2>);
        constexpr auto point2 = std::array<double, 2>{3.0 / 4, 0 };
        constexpr auto point3 = std::tuple(1.0, -3.0 / 2);

        REQUIRE(p(point1) == doctest::Approx(7.0 / 4));
        REQUIRE(p(point2) == doctest::Approx(25.0 / 16));
        REQUIRE(p(point3) == doctest::Approx(2.0));

        REQUIRE(g(point1) == doctest::Approx(7.0 / 4));
        REQUIRE(g(point2) == doctest::Approx(5.0 / 4));
        REQUIRE(g(point3) == doctest::Approx(13.0 / 4));

        constexpr auto p_plus_g = p + g;
        constexpr auto p_times_g = p * g;
        constexpr auto p_by_g = p / g;

        REQUIRE(p_plus_g(point1) == doctest::Approx(14.0 / 4));
        REQUIRE(p_plus_g(point2) == doctest::Approx(25.0 / 16 + 5.0 / 4));
        REQUIRE(p_plus_g(point3) == doctest::Approx(2.0 + 13.0 / 4));

        REQUIRE(p_times_g(point1) == doctest::Approx(49.0 / 16));
        REQUIRE(p_times_g(point2) == doctest::Approx(125.0 / 64.0));
        REQUIRE(p_times_g(point3) == doctest::Approx(26.0 / 4));

        REQUIRE(p_by_g(point1) == doctest::Approx(1.0));
        REQUIRE(p_by_g(point2) == doctest::Approx(5.0 / 4));
        REQUIRE(p_by_g(point3) == doctest::Approx(8.0 / 13));
    }
}

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End test block.
 *******************************************************************************/

/********************************************************************************
 * Test partial derivatives of a polynomial.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Polynomials] Test partial derivative computation")
{
    SUBCASE("Single variable case")
    {
        // p(x) = x^2 - 1
        constexpr auto p = Polynomial<int, Metanomials::Powers<2>, Metanomials::Powers<0>>(1, -1);
        // g(x) = x + 1;
        constexpr auto g = Polynomial<int, Metanomials::Powers<1>, Metanomials::Powers<0>>(1, 1);

        REQUIRE(p.partial<0>() == Polynomial<int, Metanomials::Powers<1>>(2));
        REQUIRE(p.partial<0>().partial<0>() == Polynomial<int, Metanomials::Powers<0>>(2));
        REQUIRE(g.partial<0>() == Polynomial<int, Metanomials::Powers<0>>(1));
        REQUIRE(g.partial<0>().partial<0>() == Polynomial<int>());

        constexpr std::array x{3.0};

        REQUIRE((p + g).partial<0>()(std::tuple(1)) == 3);
        REQUIRE((p + g).partial<0>()(std::tuple(1.5)) == doctest::Approx(4.0));
        REQUIRE((p + g).partial<0>()(x) == 7);

        REQUIRE((p * g).partial<0>()(std::tuple(1)) == 4);
        REQUIRE((p * g).partial<0>()(std::tuple(1.5)) == doctest::Approx(3 * 2.25 + 3 - 1));
        REQUIRE((p * g).partial<0>()(x) == doctest::Approx(32.0));

        REQUIRE((p / g).partial<0>()(std::tuple(1.0)) == doctest::Approx(1.0));
        REQUIRE((p / g).partial<0>()(std::tuple(1.5)) == doctest::Approx(1.0));
        REQUIRE((p / g).partial<0>()(std::tuple(3.0)) == doctest::Approx(1.0));
    }

    SUBCASE("Multivariable case")
    {
        // p(x, y) = x^2 + 2xy - 2y + 1
        constexpr auto p = Polynomial<int, Metanomials::Powers<2, 0>,
                                      Metanomials::Powers<1, 1>,
                                      Metanomials::Powers<0, 1>,
                                      Metanomials::Powers<0, 0>
                                     >(1, 2, -2, 1);
        // g(x, y) = y^2 - x + 2
        constexpr auto g = Polynomial<int, Metanomials::Powers<0, 2>,
                                      Metanomials::Powers<1, 0>, Metanomials::Powers<0, 0>
                                     >(1, -1, 2);

        // Evaluation points
        constexpr auto point1 = std::tuple(Rationals::rational<1, 2>, -Rationals::rational<1, 2>);
        constexpr auto point2 = std::array<double, 2>{3.0 / 4, 0 };
        constexpr auto point3 = std::tuple(1.0, -3.0 / 2);

        REQUIRE(partial<0>(p) == Polynomial<int, Metanomials::Powers<1, 0>,
                                            Metanomials::Powers<0, 1>>(2, 2));
        REQUIRE(partial<1>(p) == Polynomial<int, Metanomials::Powers<1, 0>,
                                            Metanomials::Powers<0, 0>>(2, -2));
        REQUIRE(partial<0>(g) == Polynomial<int, Metanomials::Powers<0, 0>>(-1));
        REQUIRE(partial<1>(g) == Polynomial<int, Metanomials::Powers<0, 1>>(2));

        REQUIRE(partial<0>(p + g)(point1) == -1);
        REQUIRE(partial<1>(p + g)(point1) == -2);
        REQUIRE(partial<0>(p + g)(point2) == doctest::Approx(0.5));
        REQUIRE(partial<1>(p + g)(point2) == doctest::Approx(-0.5));
        REQUIRE(partial<0>(p + g)(point3) == doctest::Approx(-2.0));
        REQUIRE(partial<1>(p + g)(point3) == doctest::Approx(-3.0));

        REQUIRE(partial<0>(p * g)(point1) == doctest::Approx(-7.0 / 4));
        REQUIRE(partial<1>(p * g)(point1) == doctest::Approx(-7.0 / 2));
        REQUIRE(partial<0>(p * g)(point2) == doctest::Approx(5.0 / 16));
        REQUIRE(partial<1>(p * g)(point2) == doctest::Approx(-5.0 / 8));
        REQUIRE(partial<0>(p * g)(point3) == doctest::Approx(-21.0 / 4));
        REQUIRE(partial<1>(p * g)(point3) == doctest::Approx(-6.0));

        REQUIRE(partial<0>(p / g)(point1) == doctest::Approx(4.0 / 7));
        REQUIRE(partial<1>(p / g)(point1) == doctest::Approx(0.0));
        REQUIRE(partial<0>(p / g)(point2) == doctest::Approx(11.0 / 5));
        REQUIRE(partial<1>(p / g)(point2) == doctest::Approx(-0.4));
        REQUIRE(partial<0>(p / g)(point3) == doctest::Approx(-20.0 / 169));
        REQUIRE(partial<1>(p / g)(point3) == doctest::Approx(96.0 / 169));
    }
}

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End test block.
 *******************************************************************************/

} // namespace Polynomials

} // namespace Galerkin

#endif /* POLYNOMIALS_HPP */