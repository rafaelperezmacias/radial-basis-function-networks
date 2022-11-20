package utils;

public class Functions {

    public static double sigmoid(double activationValue) {
        return 1 / ( 1 + Math.pow( Math.E, -activationValue ) );
    }

    public static double derivativeSigmoid(double activationValue) {
        return sigmoid(activationValue) * (1 - sigmoid(activationValue));
    }

    public static double relu(double activationValue) {
        return Math.max(0, activationValue);
    }

    public static double derivativeRelu(double activationValue) {
        return (activationValue >= 0) ? 1 : 0;
    }

    public static double tanh(double activationValue) {
        return (2 / (1 + Math.pow(Math.E, -2 * activationValue))) - 1;
    }

    public static double derivativeTanh(double activationValue) {
        return 1 - Math.pow(tanh(activationValue), 2);
    }

}
