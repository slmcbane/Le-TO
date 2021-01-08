/*
 * Copyright (c) 2019, The University of Texas at Austin & Sean McBane
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

/*!
 * @file Interval.hpp
 * @brief Definition of 1-dimensional interval element class
 */

#include "ElementBase.hpp"
#include "Elements.hpp"
#include "UniformScaling.hpp"

namespace Galerkin
{

namespace Elements
{

/*!
 * @brief Implements a one-dimensional interval element
 *
 * This element type has for basis functions polynomials of degree `Degree`,
 * with the degrees of freedom the values at the endpoints and equally spaced
 * points in between. Given endpoints of the interval, the instantiated element
 * is constructed as a uniform translation and scaling from the reference interval
 * [-1, 1].
 *
 * The default integration order is 2 * Degree, to integrate exactly elements
 * of a mass matrix.
 *
 * @see Galerkin::Elements::ElementBase
 */
template <int Degree, class T>
class IntervalElement : public ElementBase<IntervalElement<Degree, T>>
{
    T m_scaling;
    T m_translation;
public:
    constexpr static auto basis = derive_shape_functions(
        powers_up_to(intgr_constant<Degree>),
        evenly_spaced(Rationals::rational<-1>, Rationals::rational<1>, intgr_constant<Degree>)
            .map([](auto N) { return EvaluateAt<decltype(N)>{}; })
    );

    constexpr IntervalElement(T a, T b) : m_scaling((b-a) / 2), m_translation((a+b) / 2)
    {}

    constexpr auto coordinate_map() const noexcept
    {
        return Transforms::UniformScaling(m_scaling, std::array<T, 1>{m_translation});
    }
};

} // namespace Elements

template <int Degree, class T>
struct DefaultIntegrationOrder<Elements::IntervalElement<Degree, T>>
{
    constexpr static int order = 2 * Degree;
};

/********************************************************************************
 * Begin test block.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

namespace Elements
{

TEST_CASE("[Galerkin::Elements] Check IntervalElement basis functions")
{
    IntervalElement<1, double> elt(0.0, 1.0);

    REQUIRE(get<0>(elt.basis) ==
        Metanomials::metanomial(Metanomials::term(Rationals::rational<-1, 2>,
            Metanomials::Powers<1>{}), Metanomials::term(Rationals::rational<1, 2>,
            Metanomials::Powers<0>{})));

    IntervalElement<2, double> elt2(0.0, 1.0);

    REQUIRE(get<1>(elt2.basis) ==
        Metanomials::metanomial(
            Metanomials::term(Rationals::rational<-1>, Metanomials::Powers<2>{}),
            Metanomials::term(Rationals::rational<1>, Metanomials::Powers<0>{})
        ));
} // TEST_CASE

TEST_CASE("[Galerkin::Elements] Check mapping of points through IntervalElement")
{
    IntervalElement<1, double> elt(0.5, 1.0);

    REQUIRE(elt.transform(std::tuple(0.0))[0] == doctest::Approx(0.75));
    REQUIRE(elt.transform(std::array<float, 1>{0.5})[0] == doctest::Approx(0.875));
} // TEST_CASE

namespace IntervalTestNamespace
{

struct SymmetricMassForm
{
    template <class F, class G>
    constexpr auto operator()(const F &f, const G &g) const noexcept
    {
        return f * g;
    }
};

template <class Element>
struct SymmetricStiffnessForm
{
    const Element &el;

    constexpr SymmetricStiffnessForm(const Element &e) : el(e) {}

    template <class F, class G>
    constexpr auto operator()(const F &f, const G &g) const noexcept
    {
        return el.template partial<0>(f) * el.template partial<0>(g);
    }
};

} // namespace IntervalTestNamespace

template<>
struct IsSymmetric<IntervalTestNamespace::SymmetricMassForm> : public std::true_type
{};

template <class Element>
struct IsSymmetric<IntervalTestNamespace::SymmetricStiffnessForm<Element>> : public std::true_type
{};

TEST_CASE("[Galerkin::Elements] Test computed mass and stiffness matrices for IntervalElement")
{
    constexpr IntervalElement<2, double> elt(0.5, 1.0);

SUBCASE("Mass matrix, no symmetric tag")
{
    constexpr auto mass_matrix = elt.form_matrix(
        [](auto f, auto g) { return f * g; }
    );

    REQUIRE(mass_matrix(0, 0) == doctest::Approx(1.0 / 15));
    REQUIRE(mass_matrix(0, 1) == doctest::Approx(1.0 / 30));
    REQUIRE(mass_matrix(0, 2) == doctest::Approx(-1.0 / 60));
    REQUIRE(mass_matrix(1, 1) == doctest::Approx(4.0 / 15));
    REQUIRE(mass_matrix(1, 2) == doctest::Approx(1.0 / 30));
    REQUIRE(mass_matrix(2, 2) == doctest::Approx(1.0 / 15));
    REQUIRE(mass_matrix(0, 1) == doctest::Approx(mass_matrix(1, 0)));
    REQUIRE(mass_matrix(0, 2) == doctest::Approx(mass_matrix(2, 0)));
    REQUIRE(mass_matrix(1, 2) == doctest::Approx(mass_matrix(2, 1)));
} // SUBCASE

SUBCASE("Mass matrix, with symmetric tag")
{
    constexpr auto mass_matrix = elt.form_matrix(
        IntervalTestNamespace::SymmetricMassForm{}
    );

    REQUIRE(mass_matrix(0, 0) == doctest::Approx(1.0 / 15));
    REQUIRE(mass_matrix(0, 1) == doctest::Approx(1.0 / 30));
    REQUIRE(mass_matrix(0, 2) == doctest::Approx(-1.0 / 60));
    REQUIRE(mass_matrix(1, 1) == doctest::Approx(4.0 / 15));
    REQUIRE(mass_matrix(1, 2) == doctest::Approx(1.0 / 30));
    REQUIRE(mass_matrix(2, 2) == doctest::Approx(1.0 / 15));
    REQUIRE(mass_matrix(0, 1) == mass_matrix(1, 0));
    REQUIRE(mass_matrix(0, 2) == mass_matrix(2, 0));
    REQUIRE(mass_matrix(1, 2) == mass_matrix(2, 1));
} // SUBCASE

SUBCASE("Stiffness matrix")
{
    constexpr auto stiffness_matrix = elt.form_matrix(
        IntervalTestNamespace::SymmetricStiffnessForm(elt),
        IntegrationOrder<2>{}
    );

    REQUIRE(stiffness_matrix(0, 0) == doctest::Approx(14.0 / 3));
    REQUIRE(stiffness_matrix(0, 1) == doctest::Approx(-16.0 / 3));
    REQUIRE(stiffness_matrix(0, 2) == doctest::Approx(2.0 / 3));
    REQUIRE(stiffness_matrix(1, 1) == doctest::Approx(32.0 / 3));
    REQUIRE(stiffness_matrix(1, 2) == doctest::Approx(-16.0 / 3));
    REQUIRE(stiffness_matrix(2, 2) == doctest::Approx(14.0 / 3));
} // SUBCASE

} // TEST_CASE

} // namespace Elements

#endif // DOCTEST_LIBRARY_INCLUDED

/********************************************************************************
 * End test block
 *******************************************************************************/

} /* namespace Galerkin */

#endif /* INTERVAL_HPP */
