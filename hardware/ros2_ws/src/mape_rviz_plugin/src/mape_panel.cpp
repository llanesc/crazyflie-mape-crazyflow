#include "mape_rviz_plugin/mape_panel.hpp"

#include <QApplication>
#include <QStyle>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace mape_rviz_plugin
{

// ============================================================================
// DroneStatusWidget Implementation
// ============================================================================

DroneStatusWidget::DroneStatusWidget(const QString & name, QWidget * parent)
: QFrame(parent), is_active_(false)
{
  setFrameStyle(QFrame::Box | QFrame::Raised);
  setLineWidth(1);

  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(5, 3, 5, 3);
  layout->setSpacing(2);

  // Header row: status indicator + name
  auto * header_layout = new QHBoxLayout();
  header_layout->setSpacing(5);

  status_indicator_ = new QLabel(this);
  status_indicator_->setFixedSize(12, 12);
  status_indicator_->setStyleSheet(
    "background-color: #FF0000; border: 1px solid #333;");

  name_label_ = new QLabel(name, this);
  name_label_->setStyleSheet("font-weight: bold;");

  header_layout->addWidget(status_indicator_);
  header_layout->addWidget(name_label_);
  header_layout->addStretch();

  layout->addLayout(header_layout);

  // State labels with monospace font for alignment
  QString mono_style = "font-family: monospace; font-size: 9pt; color: #888;";

  pos_label_ = new QLabel("Pos: ---, ---, ---", this);
  pos_label_->setStyleSheet(mono_style);

  vel_label_ = new QLabel("Vel: ---, ---, ---", this);
  vel_label_->setStyleSheet(mono_style);

  att_label_ = new QLabel("RPY: ---, ---, ---", this);
  att_label_->setStyleSheet(mono_style);

  ang_vel_label_ = new QLabel("Rates: ---, ---, ---", this);
  ang_vel_label_->setStyleSheet(mono_style);

  layout->addWidget(pos_label_);
  layout->addWidget(vel_label_);
  layout->addWidget(att_label_);
  layout->addWidget(ang_vel_label_);
}

void DroneStatusWidget::setActive(bool active)
{
  is_active_ = active;
  if (active) {
    status_indicator_->setStyleSheet(
      "background-color: #00FF00; border: 1px solid #333;");
  } else {
    status_indicator_->setStyleSheet(
      "background-color: #FF0000; border: 1px solid #333;");
  }
}

void DroneStatusWidget::setName(const QString & name)
{
  name_label_->setText(name);
}

void DroneStatusWidget::setState(const multiagent_pursuit_evasion_interfaces::msg::State & state)
{
  // Position (meters)
  pos_label_->setText(QString("Pos:   %1 %2 %3 [m]")
    .arg(state.position[0], 7, 'f', 3)
    .arg(state.position[1], 7, 'f', 3)
    .arg(state.position[2], 7, 'f', 3));

  // Velocity (m/s)
  vel_label_->setText(QString("Vel:   %1 %2 %3 [m/s]")
    .arg(state.velocity[0], 7, 'f', 3)
    .arg(state.velocity[1], 7, 'f', 3)
    .arg(state.velocity[2], 7, 'f', 3));

  // Attitude (radians)
  att_label_->setText(QString("RPY:   %1 %2 %3 [rad]")
    .arg(state.attitude[0], 7, 'f', 3)
    .arg(state.attitude[1], 7, 'f', 3)
    .arg(state.attitude[2], 7, 'f', 3));

  // Angular velocity (rad/s)
  ang_vel_label_->setText(QString("Rates: %1 %2 %3 [rad/s]")
    .arg(state.angular_velocity[0], 7, 'f', 3)
    .arg(state.angular_velocity[1], 7, 'f', 3)
    .arg(state.angular_velocity[2], 7, 'f', 3));
}

// ============================================================================
// MapePanel Implementation
// ============================================================================

MapePanel::MapePanel(QWidget * parent)
: rviz_common::Panel(parent),
  node_(nullptr),
  update_timer_(nullptr)
{
  setupUI();
}

MapePanel::~MapePanel()
{
  if (update_timer_) {
    update_timer_->stop();
  }
}

void MapePanel::setupUI()
{
  main_layout_ = new QVBoxLayout(this);
  main_layout_->setSpacing(10);

  // =========================================================================
  // System Status Section
  // =========================================================================
  auto * status_group = new QGroupBox("System Status", this);
  auto * status_layout = new QHBoxLayout(status_group);

  system_status_label_ = new QLabel("Status:", this);
  status_value_label_ = new QLabel("DISCONNECTED", this);
  status_value_label_->setStyleSheet("font-weight: bold; color: gray;");

  status_layout->addWidget(system_status_label_);
  status_layout->addWidget(status_value_label_);
  status_layout->addStretch();

  main_layout_->addWidget(status_group);

  // =========================================================================
  // Control Buttons Section
  // =========================================================================
  auto * control_group = new QGroupBox("Control", this);
  auto * control_layout = new QHBoxLayout(control_group);

  initialize_button_ = new QPushButton("Initialize", this);
  initialize_button_->setMinimumHeight(40);
  initialize_button_->setEnabled(false);

  run_button_ = new QPushButton("Run", this);
  run_button_->setMinimumHeight(40);
  run_button_->setEnabled(false);
  run_button_->setStyleSheet("QPushButton:enabled { background-color: #4CAF50; color: white; }");

  off_button_ = new QPushButton("Off", this);
  off_button_->setMinimumHeight(40);
  off_button_->setEnabled(false);
  off_button_->setStyleSheet("QPushButton:enabled { background-color: #f44336; color: white; }");

  control_layout->addWidget(initialize_button_);
  control_layout->addWidget(run_button_);
  control_layout->addWidget(off_button_);

  connect(initialize_button_, &QPushButton::clicked, this, &MapePanel::onInitializeButtonClicked);
  connect(run_button_, &QPushButton::clicked, this, &MapePanel::onRunButtonClicked);
  connect(off_button_, &QPushButton::clicked, this, &MapePanel::onOffButtonClicked);

  main_layout_->addWidget(control_group);

  // =========================================================================
  // Evaders Section
  // =========================================================================
  evaders_group_ = new QGroupBox("Evaders (Blue)", this);
  evaders_group_->setStyleSheet("QGroupBox { color: #2196F3; font-weight: bold; }");
  evaders_layout_ = new QVBoxLayout(evaders_group_);
  evaders_layout_->setSpacing(5);

  main_layout_->addWidget(evaders_group_);

  // =========================================================================
  // Pursuers Section
  // =========================================================================
  pursuers_group_ = new QGroupBox("Pursuers (Red)", this);
  pursuers_group_->setStyleSheet("QGroupBox { color: #f44336; font-weight: bold; }");
  pursuers_layout_ = new QVBoxLayout(pursuers_group_);
  pursuers_layout_->setSpacing(5);

  main_layout_->addWidget(pursuers_group_);

  // Add stretch at bottom
  main_layout_->addStretch();

  // =========================================================================
  // Setup UI update timer (20Hz)
  // =========================================================================
  update_timer_ = new QTimer(this);
  connect(update_timer_, &QTimer::timeout, this, &MapePanel::updateUI);
}

void MapePanel::onInitialize()
{
  // Get the ROS node from RViz's display context
  auto ros_node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    RCLCPP_ERROR(rclcpp::get_logger("MapePanel"), "Failed to get ROS node abstraction");
    return;
  }

  node_ = ros_node_abstraction->get_raw_node();

  // Create subscription to status topic with BEST_EFFORT QoS to match publisher
  rclcpp::QoS qos(10);
  qos.best_effort();

  status_sub_ = node_->create_subscription<multiagent_pursuit_evasion_interfaces::msg::Status>(
    "/multiagent_pursuit_evasion/status",
    qos,
    std::bind(&MapePanel::statusCallback, this, std::placeholders::_1));

  // Create service client for commands
  command_client_ = node_->create_client<multiagent_pursuit_evasion_interfaces::srv::Command>(
    "/command");

  // Start UI update timer at 20Hz (50ms interval)
  update_timer_->start(50);

  RCLCPP_INFO(node_->get_logger(), "MapePanel initialized");
}

void MapePanel::statusCallback(
  const multiagent_pursuit_evasion_interfaces::msg::Status::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(status_mutex_);
  latest_status_ = msg;
}

void MapePanel::updateUI()
{
  multiagent_pursuit_evasion_interfaces::msg::Status::SharedPtr status;
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status = latest_status_;
  }

  if (!status) {
    status_value_label_->setText("DISCONNECTED");
    status_value_label_->setStyleSheet("font-weight: bold; color: gray;");
    initialize_button_->setEnabled(false);
    run_button_->setEnabled(false);
    off_button_->setEnabled(false);
    return;
  }

  // Update system status display
  switch (status->status) {
    case STATUS_OFF:
      status_value_label_->setText("OFF");
      status_value_label_->setStyleSheet("font-weight: bold; color: #666;");
      initialize_button_->setEnabled(true);
      run_button_->setEnabled(false);
      off_button_->setEnabled(false);
      break;
    case STATUS_INITIALIZED:
      status_value_label_->setText("INITIALIZED");
      status_value_label_->setStyleSheet("font-weight: bold; color: #FF9800;");
      initialize_button_->setEnabled(false);
      run_button_->setEnabled(true);  // Only enable Run when INITIALIZED
      off_button_->setEnabled(true);
      break;
    case STATUS_RUNNING:
      status_value_label_->setText("RUNNING");
      status_value_label_->setStyleSheet("font-weight: bold; color: #4CAF50;");
      initialize_button_->setEnabled(false);
      run_button_->setEnabled(false);
      off_button_->setEnabled(true);
      break;
    default:
      status_value_label_->setText("UNKNOWN");
      status_value_label_->setStyleSheet("font-weight: bold; color: red;");
  }

  // Update drone status widgets
  updateDroneStatusWidgets();
}

void MapePanel::updateDroneStatusWidgets()
{
  multiagent_pursuit_evasion_interfaces::msg::Status::SharedPtr status;
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status = latest_status_;
  }

  if (!status) return;

  // Update evader widgets - recreate if count changed
  size_t n_evaders = status->n_evaders;
  if (evader_widgets_.size() != n_evaders) {
    // Clear existing widgets
    for (auto * widget : evader_widgets_) {
      evaders_layout_->removeWidget(widget);
      delete widget;
    }
    evader_widgets_.clear();

    // Create new widgets
    for (size_t i = 0; i < n_evaders; ++i) {
      QString name = QString::fromStdString(
        i < status->evader_cf_names.size() ? status->evader_cf_names[i] : "evader_" + std::to_string(i));
      auto * widget = new DroneStatusWidget(name, evaders_group_);
      evader_widgets_.push_back(widget);
      evaders_layout_->addWidget(widget);
    }
  }

  // Update evader status and state
  for (size_t i = 0; i < evader_widgets_.size() && i < status->cf_evader_states.size(); ++i) {
    evader_widgets_[i]->setActive(status->cf_evader_states[i].active);
    evader_widgets_[i]->setState(status->cf_evader_states[i]);
    if (i < status->evader_cf_names.size()) {
      evader_widgets_[i]->setName(QString::fromStdString(status->evader_cf_names[i]));
    }
  }

  // Update pursuer widgets - recreate if count changed
  size_t n_pursuers = status->n_pursuers;
  if (pursuer_widgets_.size() != n_pursuers) {
    // Clear existing widgets
    for (auto * widget : pursuer_widgets_) {
      pursuers_layout_->removeWidget(widget);
      delete widget;
    }
    pursuer_widgets_.clear();

    // Create new widgets
    for (size_t i = 0; i < n_pursuers; ++i) {
      QString name = QString::fromStdString(
        i < status->pursuer_cf_names.size() ? status->pursuer_cf_names[i] : "pursuer_" + std::to_string(i));
      auto * widget = new DroneStatusWidget(name, pursuers_group_);
      pursuer_widgets_.push_back(widget);
      pursuers_layout_->addWidget(widget);
    }
  }

  // Update pursuer status and state
  for (size_t i = 0; i < pursuer_widgets_.size() && i < status->cf_pursuer_states.size(); ++i) {
    pursuer_widgets_[i]->setActive(status->cf_pursuer_states[i].active);
    pursuer_widgets_[i]->setState(status->cf_pursuer_states[i]);
    if (i < status->pursuer_cf_names.size()) {
      pursuer_widgets_[i]->setName(QString::fromStdString(status->pursuer_cf_names[i]));
    }
  }
}

void MapePanel::onInitializeButtonClicked()
{
  callCommandService(multiagent_pursuit_evasion_interfaces::srv::Command::Request::MAPE_CMD_INITIALIZE);
}

void MapePanel::onRunButtonClicked()
{
  callCommandService(multiagent_pursuit_evasion_interfaces::srv::Command::Request::MAPE_CMD_RUN);
}

void MapePanel::onOffButtonClicked()
{
  callCommandService(multiagent_pursuit_evasion_interfaces::srv::Command::Request::MAPE_CMD_OFF);
}

void MapePanel::callCommandService(uint8_t command)
{
  if (!command_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(node_->get_logger(), "Command service not available");
    return;
  }

  auto request = std::make_shared<multiagent_pursuit_evasion_interfaces::srv::Command::Request>();
  request->command = command;

  // Async call - we don't need to wait for response
  command_client_->async_send_request(request,
    [this](rclcpp::Client<multiagent_pursuit_evasion_interfaces::srv::Command>::SharedFuture future) {
      (void)future;  // Response is empty, just log
      RCLCPP_INFO(node_->get_logger(), "Command sent successfully");
    });
}

void MapePanel::load(const rviz_common::Config & config)
{
  Panel::load(config);
}

void MapePanel::save(rviz_common::Config config) const
{
  Panel::save(config);
}

}  // namespace mape_rviz_plugin

// Register the plugin with pluginlib
PLUGINLIB_EXPORT_CLASS(mape_rviz_plugin::MapePanel, rviz_common::Panel)

// Include MOC generated file for Q_OBJECT classes
#include "mape_rviz_plugin/moc_mape_panel.cpp"
