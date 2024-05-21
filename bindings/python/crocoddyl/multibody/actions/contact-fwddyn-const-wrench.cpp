///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/contact-fwddyn-const-wrench.hpp"

#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionContactFwdDynamicsConstWrench() {
  bp::register_ptr_to_python<
      boost::shared_ptr<DifferentialActionModelContactFwdDynamicsConstWrench> >();

  bp::class_<DifferentialActionModelContactFwdDynamicsConstWrench,
             bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelContactFwdDynamicsConstWrench",
      "Differential action model for contact forward dynamics in multibody "
      "systems.\n\n"
      "The contact is modelled as holonomic constraits in the contact frame. "
      "There\n"
      "is also a custom implementation in case of system with armatures. If "
      "you want to\n"
      "include the armature, you need to use set_armature(). On the other "
      "hand, the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<boost::shared_ptr<StateMultibody>,
               boost::shared_ptr<ActuationModelAbstract>,
               boost::shared_ptr<ContactModelMultiple>,
               boost::shared_ptr<CostModelSum>, 
               const std::vector<pinocchio::FrameIndex>&,
               const std::vector<Eigen::VectorXd>&,
               const pinocchio::ReferenceFrame,
               bp::optional<double, bool> >(
          bp::args("self", "state", "actuation", "contacts", "costs", "ids",
                   "fref", "ref_frame", "inv_damping", "enable_force"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not "
          "full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost "
          "based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. "
          "enable_force=True)."
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param contacts: multiple contact model\n"
          ":param costs: stack of cost functions\n"
          ":param inv_damping: Damping factor for cholesky decomposition of "
          "JMinvJt (default 0.)\n"
          ":param enable_force: Enable the computation of force Jacobians "
          "(default False)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>,
                    boost::shared_ptr<ActuationModelAbstract>,
                    boost::shared_ptr<ContactModelMultiple>,
                    boost::shared_ptr<CostModelSum>,
                    boost::shared_ptr<ConstraintModelManager>,
                    const std::vector<pinocchio::FrameIndex>&,
                    const std::vector<Eigen::VectorXd>&,
                    const pinocchio::ReferenceFrame,
                    bp::optional<double, bool> >(
          bp::args("self", "state", "actuation", "contacts", "costs", "constraints",
                   "ids", "fref", "ref_frame", "inv_damping", "enable_force"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not "
          "full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost "
          "based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. "
          "enable_force=True)."
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param contacts: multiple contact model\n"
          ":param costs: stack of cost functions\n"
          ":param constraints: stack of constraint functions\n"
          ":param inv_damping: Damping factor for cholesky decomposition of "
          "JMinvJt (default 0.)\n"
          ":param enable_force: Enable the computation of force Jacobians "
          "(default False)"))
      .def<void (DifferentialActionModelContactFwdDynamicsConstWrench::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelContactFwdDynamicsConstWrench::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system "
          "with contact. The\n"
          "contacts are modelled as holonomic constraints.\n"
          "Additionally it computes the cost value associated to this state "
          "and control pair.\n"
          ":param data: contact forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (DifferentialActionModelContactFwdDynamicsConstWrench::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelAbstract::calc,
          bp::args("self", "data", "x"))
      .def<void (DifferentialActionModelContactFwdDynamicsConstWrench::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelContactFwdDynamicsConstWrench::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the differential multibody system and "
          "its cost\n"
          "functions.\n\n"
          "It computes the partial derivatives of the differential multibody "
          "system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: contact forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DifferentialActionModelContactFwdDynamicsConstWrench::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelContactFwdDynamicsConstWrench::createData,
           bp::args("self"),
           "Create the contact forward dynamics differential action data.")
      .add_property(
          "pinocchio",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_pinocchio,
              bp::return_internal_reference<>()),
          "multibody model (i.e. pinocchio model)")
      .add_property(
          "actuation",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_actuation,
              bp::return_value_policy<bp::return_by_value>()),
          "actuation model")
      .add_property(
          "contacts",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_contacts,
              bp::return_value_policy<bp::return_by_value>()),
          "multiple contact model")
      .add_property("costs",
                    bp::make_function(
                        &DifferentialActionModelContactFwdDynamicsConstWrench::get_costs,
                        bp::return_value_policy<bp::return_by_value>()),
                    "total cost model")
      .add_property(
          "constraints",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_constraints,
              bp::return_value_policy<bp::return_by_value>()),
          "constraint model manager")
      .add_property(
          "armature",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_armature,
              bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::set_armature),
          "set an armature mechanism in the joints")
      .add_property(
          "JMinvJt_damping",
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::get_damping_factor),
          bp::make_function(
              &DifferentialActionModelContactFwdDynamicsConstWrench::set_damping_factor),
          "Damping factor for cholesky decomposition of JMinvJt")
      .def(CopyableVisitor<DifferentialActionModelContactFwdDynamicsConstWrench>());

  bp::register_ptr_to_python<
      boost::shared_ptr<DifferentialActionDataContactFwdDynamicsConstWrench> >();

  bp::class_<DifferentialActionDataContactFwdDynamicsConstWrench,
             bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataContactFwdDynamicsConstWrench",
      "Action data for the contact forward dynamics system.",
      bp::init<DifferentialActionModelContactFwdDynamicsConstWrench*>(
          bp::args("self", "model"),
          "Create contact forward-dynamics action data.\n\n"
          ":param model: contact forward-dynamics action model"))
      .add_property(
          "pinocchio",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::pinocchio,
                          bp::return_internal_reference<>()),
          "pinocchio data")
      .add_property(
          "multibody",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::multibody,
                          bp::return_internal_reference<>()),
          "multibody data")
      .add_property(
          "costs",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::costs,
                          bp::return_value_policy<bp::return_by_value>()),
          "total cost data")
      .add_property("constraints",
                    bp::make_getter(
                        &DifferentialActionDataContactFwdDynamicsConstWrench::constraints,
                        bp::return_value_policy<bp::return_by_value>()),
                    "constraint data")
      .add_property(
          "Kinv",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::Kinv,
                          bp::return_internal_reference<>()),
          "inverse of the KKT matrix")
      .add_property(
          "df_dx",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::df_dx,
                          bp::return_internal_reference<>()),
          "Jacobian of the contact force")
      .add_property(
          "df_du",
          bp::make_getter(&DifferentialActionDataContactFwdDynamicsConstWrench::df_du,
                          bp::return_internal_reference<>()),
          "Jacobian of the contact force")
      .def(CopyableVisitor<DifferentialActionDataContactFwdDynamicsConstWrench>());
}

}  // namespace python
}  // namespace crocoddyl
