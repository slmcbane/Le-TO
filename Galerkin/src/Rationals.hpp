/*
 * Copyright (c) 2019, Sean McBane
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef RATIONALS_HPP
#define RATIONALS_HPP

/*!
 * @file Rationals.hpp
 * @brief Compile-time computations with rational numbers.
 * @author Sean McBane <sean.mcbane@protonmail.com>
 */

#include <cstdint>
#include <type_traits>

#include "utils.hpp"

namespace Galerkin
{

/// The `Rationals` namespace encapsulates all of the functionality for rational numbers
namespace Rationals
{

/*!
 * @brief The type used for the numerator in a rational. On gcc or clang this
 * could be made a int128_t, but in practice compiler limits will be reached
 * before overflow, anyway. Should be a signed type.
 */
typedef int64_t rational_num_t;

/// The type used for the denominator in a rational. Should be an unsigned type.
typedef uint64_t rational_den_t;

/*!
 * @brief Type representing a rational number at compile time.
 * 
 * `Rational` is used to do exact calculations of derivatives, Jacobians, etc.
 * by manipulating rational numbers at compile time. The normal arithmetic
 * operators are all defined for it, and it decays to a `double` as appropriate.
 * Rather than use this class template, however, you should use the `rational`
 * template constant from this header; instantiating `rational<N, D>`
 * automatically reduces the resulting fraction to its simplest form.
 */
template <rational_num_t Num, rational_den_t Den>
struct Rational
{
    static_assert(Den != zero<rational_den_t>);

    /// Get the numerator
    static constexpr auto num() { return Num; }
    /// Get the denominator
    static constexpr auto den() { return Den; }

    /*! @brief Convert the number to a double precision float, for when you have to
     * do inexact floating point operations or run-time computation :(.
     */
    constexpr operator double() const
    {
        return static_cast<double>(Num) / Den;
    }
};

template <class T>
constexpr bool is_rational = false;

template <rational_num_t N, rational_den_t D>
constexpr bool is_rational<Rational<N, D>> = true;

/*!
 * @brief Utility; find greatest common denominator of `a` and `b`. This will hit
 * `constexpr` evaluation limits when `a` or `b` becomes large relative to the
 * other.
 */
constexpr auto gcd(rational_num_t a, rational_den_t b)
{
    if (a < 0)
    {
        a = -a;
    }

    auto x = static_cast<rational_den_t>(a);
    auto y = x > b ? b : x;
    x = x > b ? x : b;

    if (y == 0)
    {
        return x;
    }

    while (x != y)
    {
        x = x - y;
        if (y > x)
        {
            auto tmp = y;
            y = x;
            x = tmp;
        }
    }

    return x;
}

/// Reduce a rational number to its simplest representation.
template <rational_num_t N, rational_den_t D>
constexpr auto reduce_rational(Rational<N, D>)
{
    constexpr auto div = gcd(N, D);
    return Rational<N / static_cast<rational_num_t>(div), D / div>();
}

/*!
 * @brief Template constant for a compile-time rational.
 * 
 * This constant evaluates to a `Rational` reduced to its lowest terms. Default
 * denominator is 1, so that an integer can be constructed by `rational<n>`.
 * 
 * @tparam N The numerator
 * @tparam D The denominator. Default value: 1
 */
template <rational_num_t N, rational_den_t D = 1>
constexpr auto rational = reduce_rational(Rational<N, D>());

template <auto N1, auto D1, auto N2, auto D2>
constexpr bool operator==(Rational<N1, D1>, Rational<N2, D2>)
{
    return std::is_same_v<decltype(rational<N1, D1>), decltype(rational<N2, D2>)>;
}

/********************************************************************************
 * Tests of rational construction
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Rationals] Testing construction of rationals")
{
    REQUIRE(rational<1> == rational<2, 2>);
    REQUIRE(rational<-2, 2> == rational<-42, 42>);
    REQUIRE(rational<2, 4> == rational<1, 2>);
    REQUIRE(rational<4, 2> == rational<2, 1>);
    REQUIRE(rational<-1, 3> == rational<-6, 18>);
    // This should trigger a static assert
    // REQUIRE(rational<1, 0> == rational<0, 1>);
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

template <auto N1, auto D1, auto N2, auto D2>
constexpr auto operator+(Rational<N1, D1>, Rational<N2, D2>)
{
    constexpr auto lcm = D1 * D2 / gcd(D1, D2);
    constexpr auto mult1 = static_cast<rational_num_t>(lcm / D1);
    constexpr auto mult2 = static_cast<rational_num_t>(lcm / D2);

    return rational<N1*mult1 + N2*mult2, D1*mult1>;
}

/********************************************************************************
 * Tests of rational addition
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Rationals] Testing rational addition")
{
    REQUIRE(rational<1> + rational<2> == rational<3>);
    REQUIRE(rational<1, 2> + rational<1, 3> == rational<5, 6>);
    REQUIRE(rational<5, 6> + rational<1, 6> == rational<1>);
    REQUIRE(rational<5, 8> + rational<22, 16> == rational<2>);
    REQUIRE(rational<1, 3> + rational<3, 1> == rational<10, 3>);
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

template <auto N1, auto D1>
constexpr auto operator-(Rational<N1, D1>)
{
    return rational<-N1, D1>;
}

template <auto N1, auto D1, auto N2, auto D2>
constexpr auto operator-(Rational<N1, D1>, Rational<N2, D2>)
{
    return Rational<N1, D1>() + (-Rational<N2, D2>());
}

/********************************************************************************
 * Tests of rational subtraction and negation.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Rationals] Testing rational subtraction")
{
    REQUIRE(rational<1> - rational<2> == rational<-1>);
    REQUIRE(rational<1, 2> - rational<1, 3> == rational<1, 6>);
    REQUIRE(rational<5, 6> - rational<1, 6> == rational<2, 3>);
    REQUIRE(rational<5, 8> - rational<22, 16> == rational<-3, 4>);
    REQUIRE(rational<1, 3> - rational<3, 1> == rational<-8, 3>);
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

template <auto N1, auto D1, auto N2, auto D2>
constexpr auto operator*(Rational<N1, D1>, Rational<N2, D2>)
{
    return rational<N1*N2, D1*D2>;
}

template <auto N1, auto D1, auto N2, auto D2>
constexpr auto operator/(Rational<N1, D1>, Rational<N2, D2>)
{
    constexpr auto num = N1 * static_cast<rational_num_t>(D2);
    if constexpr (N2 < 0)
    {
        return -rational<num, D1 * static_cast<rational_den_t>(-N2)>;
    }
    else
    {
        return rational<num, D1 * static_cast<rational_den_t>(N2)>;
    }
}

/// A `Rational` * an `integral_constant` returns a `Rational`.
template <auto N, auto D, class I, I v>
constexpr auto operator*(Rational<N, D>, std::integral_constant<I, v>) noexcept
{
    return rational<N, D> * rational<v>;
}

template <auto N, auto D, class I, I v>
constexpr auto operator*(std::integral_constant<I, v>, Rational<N, D>) noexcept
{
    return rational<N, D> * rational<v>;
}

/// A `Rational` / an `integral_constant` returns a `Rational`.
template <auto N, auto D, class I, I v>
constexpr auto operator/(Rational<N, D>, std::integral_constant<I, v>)
{
    static_assert(v != 0);
    if constexpr (v < 0)
    {
        return -(rational<N, D> / std::integral_constant<I, -v>());
    }
    else
    {
        return rational<N, D> / rational<v>;
    }
}

template <auto N, auto D, class T>
constexpr auto operator*(Rational<N, D>, T x)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return (N * x) / D;
    }
    else
    {
        return (N * static_cast<double>(x)) / D;
    }
}

template <auto N, auto D, class T>
constexpr auto operator/(Rational<N, D>, T x)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return N / (D * x);
    }
    else
    {
        return N / (D * static_cast<double>(x));
    }
}

/********************************************************************************
 * Tests of rational multiplication and division.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Rationals] Testing rational multiplication and division")
{
    REQUIRE(rational<1, 2> * rational<1, 2> == rational<1, 4>);
    REQUIRE(rational<1, 2> * rational<1, 3> == rational<1, 3> * rational<1, 2>);
    REQUIRE(rational<1, 2> * rational<1, 3> == rational<1, 6>);
    REQUIRE(rational<3, 10> * rational<1, 3> == rational<1, 10>);
    REQUIRE(rational<3, 10> * rational<-1, 3> == rational<-1, 10>);

    REQUIRE(rational<1, 2> / rational<1, 2> == rational<1>);
    REQUIRE(rational<1, 2> / rational<2> == rational<1, 4>);
    REQUIRE(rational<3, 10> / rational<1, 3> == rational<9, 10>);
    REQUIRE(rational<1, 6> / rational<1, 3> == rational<1, 2>);
    REQUIRE(rational<1, 6> / rational<1, 2> == rational<1, 3>);
    REQUIRE(rational<3, 10> / rational<-1, 3> == rational<-9, 10>);

    REQUIRE(rational<3, 10> / std::integral_constant<int, 3>() == rational<1, 10>);
    REQUIRE(rational<1, 6> * std::integral_constant<int, 3>() == rational<1, 2>);
    REQUIRE(rational<1, 6> / std::integral_constant<int, -2>() == rational<-1, 12>);

    REQUIRE(rational<1, 2> * 0.5 == doctest::Approx(0.25));
    REQUIRE(rational<1, 2> / 3 == doctest::Approx(1.0 / 6));
    REQUIRE(std::is_same_v<
        decltype(rational<1, 2> * std::declval<float>()), float>);
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/


} // namespace Rationals

} // namespace Galerkin

#endif /* RATIONALS_HPP */