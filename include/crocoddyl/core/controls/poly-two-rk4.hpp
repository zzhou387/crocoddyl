///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree two, that is a quadratic function
 *
 * The size of the parameters \f$\mathbf{p}\f$ is 3 times the size of the control input \f$\mathbf{u}\f$.
 * The first third of \f$\mathbf{p}\f$ represents the value of \f$\mathbf{u}\f$ at time 0. The second
 * third of \f$\mathbf{p}\f$ represents the value of \f$\mathbf{u}\f$ at time 0.5. The last third of
 * \f$\mathbf{p}\f$ represents the value of \f$\mathbf{u}\f$ at time 1. This parametrization is suitable
 * to be used with the RK-4 integration scheme, because it requires the value of \f$\mathbf{u}\f$ exactly
 * at 0, 0.5 and 1.
 *
 * The main computations are carrying out in `calc`, `multiplyByJacobian` and `multiplyJacobianTransposeBy`,
 * where the former computes control input \f$\mathbf{w}\in\mathbb{R}^{nw}\f$ from a set of control parameters
 * \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ where `nw` and `nu` represent the dimension of the control inputs and parameters,
 * respectively, and the latter defines useful operations across the Jacobian of the control-parametrization model.
 * Finally, `params` allows us to obtain the control parameters from a the control input, i.e., it is the
 * dual of `calc`.
 * Note that `multiplyByJacobian` and `multiplyJacobianTransposeBy` requires to run `calc` first.
 *
 * \sa `ControlParametrizationAbstractTpl`, `calc()`, `calcDiff()`, `createData()`, `params`, `multiplyByJacobian`,
 * `multiplyJacobianTransposeBy`
 */
template <typename _Scalar>
class ControlParametrizationModelPolyTwoRK4Tpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;
  typedef ControlParametrizationDataPolyTwoRK4Tpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the poly-two RK4 control parametrization
   *
   * @param[in] nw  Dimension of control vector
   */
  explicit ControlParametrizationModelPolyTwoRK4Tpl(const std::size_t nw);
  virtual ~ControlParametrizationModelPolyTwoRK4Tpl();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                    const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                        const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Create the control-parametrization data
   *
   * @return the control-parametrization data
   */
  virtual boost::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                      const Eigen::Ref<const VectorXs>& w) const;

  /**
   * @brief Map the specified bounds from the control space to the parameter space
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control lower bound
   * @param[out] u_lb   Control parameters lower bound
   * @param[out] u_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& w_lb, const Eigen::Ref<const VectorXs>& w_ub,
                             Eigen::Ref<VectorXs> u_lb, Eigen::Ref<VectorXs> u_ub) const;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the
   * parameters)
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   * @param[in] op      Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyByJacobian(const boost::shared_ptr<ControlParametrizationDataAbstract>& data,
                                  const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
                                  const AssignmentOp = setto) const;

  /**
   * @brief Compute the product between the transposed Jacobian of the control (with respect to the parameters) and
   * a specified matrix
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control with respect to the parameters and the
   * matrix A
   * @param[in] op      Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyJacobianTransposeBy(const boost::shared_ptr<ControlParametrizationDataAbstract>& data,
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
                                           const AssignmentOp = setto) const;

 protected:
  using Base::nu_;
  using Base::nw_;
};

template <typename _Scalar>
struct ControlParametrizationDataPolyTwoRK4Tpl : public ControlParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector3s Vector3s;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataPolyTwoRK4Tpl(Model<Scalar>* const model) : Base(model), tmp_t2(0.) {
    c.setZero();
  }

  virtual ~ControlParametrizationDataPolyTwoRK4Tpl() {}

  Vector3s c;     //!< Polynomial coefficients of the second-order control model that depends on time
  Scalar tmp_t2;  //!< Temporary variable to store the square of the time
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-two-rk4.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
