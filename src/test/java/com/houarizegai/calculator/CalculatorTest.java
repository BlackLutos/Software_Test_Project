package com.houarizegai.calculator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CalculatorTest {

    private Calculator calculator;

    @BeforeEach
    void setUp() { // Create object before compilation
        calculator = new Calculator();
    }

    /*
     * testCalc() test method
     */
    @Test
    void testCalc() {
        double first = 3;
        String second = "5";

        try {
            Assertions.assertEquals(8, calculator.calc(first, second, '+'));
            Assertions.assertEquals(-2, calculator.calc(first, second, '-'));
            Assertions.assertEquals(15, calculator.calc(first, second, '*'));
            Assertions.assertEquals(0.6, calculator.calc(first, second, '/'));
            Assertions.assertEquals(3d, calculator.calc(first, second, '%'));
            Assertions.assertEquals(243, calculator.calc(first, second, '^'));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    @Test
    void testCalc_plus(){
        double first = 200;
        String second = "100";
        double answer = 200 + 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '+'));
    }
    @Test
    void testCalc_minus(){
        double first = 200;
        String second = "100";
        double answer = 200 - 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '-'));
    }
    @Test
    void testCalc_multiplication(){
        double first = 200;
        String second = "100";
        double answer = 200 * 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '*'));
    }
    @Test
    void testCalc_division(){
        double first = 200;
        String second = "100";
        double answer = 200 / 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '/'));
    }
    @Test
    void testCalc_modulo(){
        double first = 200;
        String second = "100";
        double answer = 200 % 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '%'));
    }
    @Test
    void testCalc_square(){
        double first = 200;
        String second = "100";
        double answer = Math.pow(200,100);
        Assertions.assertEquals(answer, calculator.calc(first, second, '^'));
    }
    @Test
    void testCalc_sqrt(){
        double first = 3;
        String second = "10";
        double answer = Math.sqrt(10);
        Assertions.assertEquals(answer, calculator.calc(first, second, '√'));
    }
    @Test
    void testCalc_ln(){
        double first = 3;
        String second = "10";
        double answer = Math.log(10);
        Assertions.assertEquals(answer, calculator.calc(first, second, 'l'));
    }
    @Test
    void getText(){
        calculator.setVal("5");
        Assertions.assertEquals("5",calculator.getResult());
        System.out.println(calculator.getResult());
    }
}
