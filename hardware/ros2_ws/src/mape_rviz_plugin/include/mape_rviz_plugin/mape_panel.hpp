#ifndef MAPE_RVIZ_PLUGIN__MAPE_PANEL_HPP_
#define MAPE_RVIZ_PLUGIN__MAPE_PANEL_HPP_

#include <memory>
#include <mutex>
#include <vector>
#include <string>

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QGroupBox>
#include <QTimer>
#include <QFrame>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include "multiagent_pursuit_evasion_interfaces/msg/status.hpp"
#include "multiagent_pursuit_evasion_interfaces/msg/evader_state.hpp"
#include "multiagent_pursuit_evasion_interfaces/msg/pursuer_state.hpp"
#include "multiagent_pursuit_evasion_interfaces/srv/command.hpp"

namespace mape_rviz_plugin
{

class DroneStatusWidget : public QFrame
{
public:
  explicit DroneStatusWidget(const QString & name, QWidget * parent = nullptr);
  void setActive(bool active);
  void setName(const QString & name);
  void setState(const multiagent_pursuit_evasion_interfaces::msg::EvaderState & state);
  void setState(const multiagent_pursuit_evasion_interfaces::msg::PursuerState & state);

private:
  QLabel * name_label_;
  QLabel * status_indicator_;
  bool is_active_;

  // State display labels
  QLabel * pos_label_;
  QLabel * vel_label_;
  QLabel * att_label_;
  QLabel * ang_vel_label_;
};

class MapePanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit MapePanel(QWidget * parent = nullptr);
  ~MapePanel() override;

  void onInitialize() override;
  void load(const rviz_common::Config & config) override;
  void save(rviz_common::Config config) const override;

protected Q_SLOTS:
  void onTakeoffButtonClicked();
  void onRunButtonClicked();
  void onOffButtonClicked();
  void updateUI();

private:
  void setupUI();
  void statusCallback(const multiagent_pursuit_evasion_interfaces::msg::Status::SharedPtr msg);
  void updateDroneStatusWidgets();
  void callCommandService(uint8_t command);

  // ROS2 members
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<multiagent_pursuit_evasion_interfaces::msg::Status>::SharedPtr status_sub_;
  rclcpp::Client<multiagent_pursuit_evasion_interfaces::srv::Command>::SharedPtr command_client_;

  // Latest status message (thread-safe access via mutex)
  std::mutex status_mutex_;
  multiagent_pursuit_evasion_interfaces::msg::Status::SharedPtr latest_status_;

  // UI elements
  QVBoxLayout * main_layout_;

  // System status section
  QLabel * system_status_label_;
  QLabel * status_value_label_;
  QLabel * solver_status_label_;
  QLabel * solver_value_label_;

  // Control buttons
  QPushButton * takeoff_button_;
  QPushButton * run_button_;
  QPushButton * off_button_;

  // Drone status sections
  QGroupBox * evaders_group_;
  QGroupBox * pursuers_group_;
  QVBoxLayout * evaders_layout_;
  QVBoxLayout * pursuers_layout_;

  // Dynamic drone status widgets
  std::vector<DroneStatusWidget*> evader_widgets_;
  std::vector<DroneStatusWidget*> pursuer_widgets_;

  // UI update timer (50ms = 20Hz)
  QTimer * update_timer_;

  // Status constants (must match Status.msg)
  static constexpr uint8_t STATUS_OFF = 0;
  static constexpr uint8_t STATUS_TAKEOFF = 1;
  static constexpr uint8_t STATUS_INITIALIZED = 2;
  static constexpr uint8_t STATUS_RUNNING = 3;
  static constexpr uint8_t STATUS_BLUE_WON = 4;
  static constexpr uint8_t STATUS_RED_WON = 5;
};

}  // namespace mape_rviz_plugin

#endif  // MAPE_RVIZ_PLUGIN__MAPE_PANEL_HPP_
