from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

def get_ros_param(node: Node, param_name: str, param_type, default_value=None, description="", additional_constraints="", read_only=False):
    # Declare the parameter with the given attributes
    param_config = ParameterDescriptor()
    param_config.description = description
    param_config.additional_constraints = additional_constraints
    param_config.read_only = read_only
    node.declare_parameter(param_name, value=default_value, descriptor=param_config)

    # Check if the parameter exists and handle its value based on the type
    try:
        param_value = node.get_parameter(param_name).get_parameter_value()
        if param_type == int:
            return param_value.integer_value
        elif param_type == float:
            return param_value.double_value
        elif param_type == str:
            return param_value.string_value
        elif param_type == bool:
            return param_value.bool_value
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    except Exception as e:
        # Handle missing parameter or other errors gracefully
        print(f"Error getting parameter {param_name}: {e}")
        return default_value  # Return default value if error occurs