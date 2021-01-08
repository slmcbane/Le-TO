#ifndef LEGENDRE_HPP
#define LEGENDRE_HPP

#include "Metanomials.hpp"

#include <array>
#include <limits>

namespace Galerkin
{

namespace Legendre
{

// This works up to 14th order before hitting a constexpr evaluation limit in
// clang.
template <auto I>
constexpr inline auto polynomial = 
    (Metanomials::metanomial(
        term(Rationals::rational<2*I-1>,
             Metanomials::powers(intgr_constant<1>))) * polynomial<I-1>
    - polynomial<I-2> * intgr_constant<I-1>) /
    intgr_constant<I>;

template <>
constexpr inline auto polynomial<0> = Metanomials::metanomial(
    term(Rationals::rational<1>,
         Metanomials::powers(intgr_constant<0>))
);

template <>
constexpr inline auto polynomial<1> = Metanomials::metanomial(
    term(Rationals::rational<1>,
         Metanomials::powers(intgr_constant<1>))
);

/********************************************************************************
 * Test that Legendre polynomials are correct.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

using namespace Metanomials;
using namespace Rationals;

TEST_CASE("Test computed Legendre polynomials")
{
    REQUIRE(polynomial<0> == Metanomials::metanomial(term(rational<1>, powers(intgr_constant<0>))));
    REQUIRE(polynomial<1> == Metanomials::metanomial(term(rational<1>, powers(intgr_constant<1>))));
    REQUIRE(polynomial<2> ==
        Metanomials::metanomial(
            term(rational<3, 2>, powers(intgr_constant<2>)),
            term(-rational<1, 2>, powers(intgr_constant<0>))
        )
    );

    REQUIRE(polynomial<10> ==
        metanomial(
            term(rational<46189, 256>, powers(intgr_constant<10>)),
            term(-rational<109395, 256>, powers(intgr_constant<8>)),
            term(rational<90090, 256>, powers(intgr_constant<6>)),
            term(-rational<30030, 256>, powers(intgr_constant<4>)),
            term(rational<3465, 256>, powers(intgr_constant<2>)),
            term(-rational<63, 256>, powers(intgr_constant<0>))
        )
    );
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

template <class T, class P>
constexpr auto interval_root(P poly, T low, T high)
{
    constexpr auto pprime = P::template partial<0>();
    T mid = (high + low) / 2;
    constexpr auto abs = [] (auto x) { return x < 0 ? -x : x; };
    while (poly(mid) != zero<T>)
    {
        auto delta = -poly(mid) / pprime(mid);
        if (abs(delta / mid) < 4 * std::numeric_limits<T>::epsilon())
        {
            break;
        }
        mid += delta;
    }
    return mid;
}

template <class T, auto Order>
constexpr auto all_roots()
{
    if constexpr (Order == 1)
    {
        return std::array<T, 1> { 0 };
    }
    else
    {
        auto extrema = all_roots<T, Order-1>();
        std::array<T, Order> my_roots { 0 };
        my_roots[0] = interval_root(polynomial<Order>, -one<T>, extrema[0]);
        for (int i = 1; i < Order-1; ++i)
        {
            my_roots[i] = interval_root(polynomial<Order>, extrema[i-1], extrema[i]);
        }
        my_roots[Order-1] = interval_root(polynomial<Order>, extrema[Order-2], one<T>);
        return my_roots;
    }
}

// This works up to order 6 before hitting a constexpr evaluation limit for my
// tested version of clang.
template <class T, auto Order>
constexpr auto roots = all_roots<T, Order>();

/********************************************************************************
 * Test rootfinding of Legendre polynomials.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

#include <cmath>

TEST_CASE("Find roots of Legendre polynomials")
{
    REQUIRE(roots<double, 1>[0] == doctest::Approx(0.0));

    REQUIRE(roots<double, 2>[0] == doctest::Approx(-1 / std::sqrt(3)));
    REQUIRE(roots<double, 2>[1] == doctest::Approx(1 / std::sqrt(3)));

    REQUIRE(roots<double, 3>[0] == doctest::Approx(-std::sqrt(3.0 / 5)));
    REQUIRE(roots<double, 3>[1] == doctest::Approx(0.0));
    REQUIRE(roots<double, 3>[2] == doctest::Approx(std::sqrt(3.0 / 5)));

    REQUIRE(roots<double, 5>[0] == doctest::Approx(
        -std::sqrt(5 + 2 * std::sqrt(10.0 / 7)) / 3
    ));
    REQUIRE(roots<double, 5>[1] == doctest::Approx(
        -std::sqrt(5 - 2 * std::sqrt(10.0 / 7)) / 3
    ));
    REQUIRE(roots<double, 5>[2] == doctest::Approx(0.0));
    REQUIRE(roots<double, 5>[3] == doctest::Approx(
        std::sqrt(5 - 2 * std::sqrt(10.0 / 7)) / 3
    ));
    REQUIRE(roots<double, 5>[4] == doctest::Approx(
        std::sqrt(5 + 2 * std::sqrt(10.0 / 7)) / 3
    ));
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/
} /* namespace Legendre */

} /* namespace Galerkin */

#endif /* LEGENDRE_HPP */