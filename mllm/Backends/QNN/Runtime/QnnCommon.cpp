/**
 * @file QnnCommon.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnCommon.hpp"
#include "mllm/Utils/Common.hpp"
#include <cstring>

namespace mllm::qnn {

// The QnnPerf is ref from MNN's impl:
// MNN/source/backend/qnn/backend/QnnPerf.hpp
QnnPerf::QnnPerf(const QNN_INTERFACE_VER_TYPE* qnn_interface) : qnn_interface_(qnn_interface) {
  QnnDevice_Infrastructure_t device_infra = nullptr;
  qnn_interface->deviceGetInfrastructure(&device_infra);
  QnnHtpDevice_Infrastructure_t* htp_infra =
      static_cast<QnnHtpDevice_Infrastructure_t*>(device_infra);
  perf_infra_ = htp_infra->perfInfra;

  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  perf_infra_.createPowerConfigId(deviceId, coreId, &power_cfg_id_);

  // clang-format off
  power_cfg_burst_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = power_cfg_id_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 0,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 40,  // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1, // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 1,  // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,  // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          },
  };

  power_cfg_balanced_ = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = power_cfg_id_,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 1,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 1000,  // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,  // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,  // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
          },
  };
  // clang-format on
}

QnnPerf::~QnnPerf() { perf_infra_.destroyPowerConfigId(power_cfg_id_); }

void QnnPerf::setRpcLatencyAndPolling() {
  // set RPC Control Latency
  QnnHtpPerfInfrastructure_PowerConfig_t rpc_ctrl_latency;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpc_ctrl_latency, 0, sizeof(rpc_ctrl_latency));

  rpc_ctrl_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  // use rpc control latency recommended 100 us, refer hexagon sdk
  rpc_ctrl_latency.rpcControlLatencyConfig = 100;
  const QnnHtpPerfInfrastructure_PowerConfig_t* power_cfg_0[] = {&rpc_ctrl_latency, nullptr};

  // set RPC latency config on power config ID created
  perf_infra_.setPowerConfig(power_cfg_id_, power_cfg_0);

  // set RPC Polling
  QnnHtpPerfInfrastructure_PowerConfig_t rpc_rolling_time;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpc_rolling_time, 0, sizeof(rpc_rolling_time));
  rpc_rolling_time.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpc_rolling_time.rpcPollingTimeConfig = 9999;  // use rpc polling time recommended 0-10000 us
  const QnnHtpPerfInfrastructure_PowerConfig_t* power_cfg_1[] = {&rpc_rolling_time, nullptr};

  // set RPC polling config on power config ID created
  perf_infra_.setPowerConfig(power_cfg_id_, power_cfg_1);
}

void QnnPerf::setPowerConfigBurst() {
  MLLM_INFO("Switching to burst mode feels like Misaka unleashing her Railgun, unrestrained "
            "power surging forward, unstoppable and electrifying!");
  const QnnHtpPerfInfrastructure_PowerConfig_t* power_cfg[] = {&power_cfg_burst_, nullptr};
  perf_infra_.setPowerConfig(power_cfg_id_, power_cfg);
}

void QnnPerf::setPowerConfigBalanced() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* power_cfg[] = {&power_cfg_balanced_, nullptr};
  perf_infra_.setPowerConfig(power_cfg_id_, power_cfg);
}

}  // namespace mllm::qnn