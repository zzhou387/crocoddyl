#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn-const-wrench.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsConstWrenchTpl(
        boost::shared_ptr<StateMultibody> state,
        boost::shared_ptr<ActuationModelAbstract> actuation,
        boost::shared_ptr<ContactModelMultiple> contacts,
        boost::shared_ptr<CostModelSum> costs, 
        const std::vector<pinocchio::FrameIndex>& ids, 
        const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& external_wrenches,
        const pinocchio::ReferenceFrame type,
        const Scalar JMinvJt_damping,
        const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(nullptr),
      pinocchio_(*state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force),
      ids_(ids),
      external_wrenches_(external_wrenches),
      type_(type){
  init();
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsConstWrenchTpl(
        boost::shared_ptr<StateMultibody> state,
        boost::shared_ptr<ActuationModelAbstract> actuation,
        boost::shared_ptr<ContactModelMultiple> contacts,
        boost::shared_ptr<CostModelSum> costs,
        boost::shared_ptr<ConstraintModelManager> constraints,
        const std::vector<pinocchio::FrameIndex>& ids, 
        const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& external_wrenches,
        const pinocchio::ReferenceFrame type,
        const Scalar JMinvJt_damping, const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), constraints->get_ng(),
           constraints->get_nh()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force),
      ids_(ids),
      external_wrenches_(external_wrenches),
      type_(type) {
  init();
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<
    Scalar>::~DifferentialActionModelContactFwdDynamicsConstWrenchTpl() {}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::init() {
  if (JMinvJt_damping_ < Scalar(0.)) {
    JMinvJt_damping_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive, set to 0");
  }
  if (contacts_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Contacts doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }

  const std::size_t nc = contacts_->get_nc();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by
  // the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_->calc(d->multibody.actuation, x, u);
  contacts_->calc(d->multibody.contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.contacts->Jc.topRows(nc));

  if (Jc_lu.rank() < d->multibody.contacts->Jc.topRows(nc).rows() &&
      JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty(
        "A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif

  // Compute the Jacobians of the external contacts
  MatrixXs Jext = MatrixXs::Zero(6 * ids_.size(), state_->get_nv());
  VectorXs fext = VectorXs::Zero(state_->get_nv());
  for (std::size_t i = 0; i < ids_.size(); ++i) {
    pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, ids_[i], type_, Jext.block(6 * i, 0, 6, state_->get_nv()));
    fext += Jext.block(6 * i, 0, 6, state_->get_nv()).transpose() * external_wrenches_[i];
  }
  pinocchio::forwardDynamics(
      pinocchio_, d->pinocchio, d->multibody.actuation->tau,
      d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
      fext, JMinvJt_damping_);
  // pinocchio::forwardDynamics(
  //     pinocchio_, d->pinocchio, d->multibody.actuation->tau,
  //     d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
  //     JMinvJt_damping_);
  d->xout = d->pinocchio.ddq;
  contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);
  d->multibody.joint->a = d->pinocchio.ddq;
  d->multibody.joint->tau = u;
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }

  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  // Computing the dynamics derivatives
  // We resize the Kinv matrix because Eigen cannot call block operations
  // recursively: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=408. Therefore,
  // it is not possible to pass d->Kinv.topLeftCorner(nv + nc, nv + nc)
  d->Kinv.resize(nv + nc, nv + nc);

  // Update the external force at each joint with the constant wrench
  for (size_t id = 0; id < ids_.size(); id++) {
    const pinocchio::JointIndex joint =
      pinocchio_.frames[id].parent;
    
    // Compute the external force at the joint based on the reference type
    const VectorXs wrench_i = external_wrenches_[id];
    Force f_local, f_ext;
    switch (type_) {
      case pinocchio::ReferenceFrame::LOCAL:
        f_local = Force(wrench_i);
        f_ext = pinocchio_.frames[id].placement.act(f_local);
        break;
      case pinocchio::ReferenceFrame::WORLD:
        throw std::invalid_argument("The WORLD reference frame is not supported yet");
        break;
      case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
        const Eigen::Ref<const Matrix3s> oRf = d->pinocchio.oMf[id].rotation();
        f_local.linear().noalias() = oRf.transpose() * wrench_i.head(3);
        f_local.angular().noalias() = oRf.transpose() * wrench_i.tail(3);
        f_ext = pinocchio_.frames[id].placement.act(f_local);
        break;
    }
    // Overwrite the forces saved in the contact data
    // Warning: make sure the external contact frames and the frames computed by the
    // contact model are different
    d->multibody.contacts->fext[joint] = f_ext;
  }
  

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout,
                                    d->multibody.contacts->fext);
  contacts_->updateRneaDiff(d->multibody.contacts, d->pinocchio);
  pinocchio::getKKTContactDynamicMatrixInverse(
      pinocchio_, d->pinocchio, d->multibody.contacts->Jc.topRows(nc), d->Kinv);

  actuation_->calcDiff(d->multibody.actuation, x, u);
  contacts_->calcDiff(d->multibody.contacts, x);

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  const Eigen::Block<MatrixXs> f_partial_dtau =
      d->Kinv.bottomLeftCorner(nc, nv);
  const Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= a_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
  d->Fx.noalias() += a_partial_dtau * d->multibody.actuation->dtau_dx;
  d->Fu.noalias() = a_partial_dtau * d->multibody.actuation->dtau_du;
  d->multibody.joint->da_dx = d->Fx;
  d->multibody.joint->da_du = d->Fu;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.topLeftCorner(nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.topRightCorner(nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.topRows(nc).noalias() +=
        f_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
    d->df_dx.topRows(nc).noalias() -=
        f_partial_dtau * d->multibody.actuation->dtau_dx;
    d->df_du.topRows(nc).noalias() =
        -f_partial_dtau * d->multibody.actuation->dtau_du;
    contacts_->updateAccelerationDiff(d->multibody.contacts,
                                      d->Fx.bottomRows(nv));
    contacts_->updateForceDiff(d->multibody.contacts, d->df_dx.topRows(nc),
                               d->df_du.topRows(nc));
  }
  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  costs_->calcDiff(d->costs, x);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x, std::size_t,
    Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  DifferentialActionDataContactFwdDynamicsConstWrenchTpl<Scalar>* d =
      static_cast<DifferentialActionDataContactFwdDynamicsConstWrenchTpl<Scalar>*>(
          data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();

  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
                  d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic);

  // Allocates memory
  d->tmp_Jstatic.conservativeResize(nv, nu_ + nc);
  d->tmp_Jstatic.leftCols(nu_) = d->multibody.actuation->dtau_du;
  d->tmp_Jstatic.rightCols(nc) =
      d->multibody.contacts->Jc.topRows(nc).transpose();
  u.noalias() = (pseudoInverse(d->tmp_Jstatic) * d->pinocchio.tau).head(nu_);
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
bool DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelContactFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", nc=" << contacts_->get_nc() << "}";
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_ng()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_nh()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<ContactModelMultipleTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
const Scalar DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::set_armature(
    const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsConstWrenchTpl<Scalar>::set_damping_factor(
    const Scalar damping) {
  if (damping < 0.) {
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
