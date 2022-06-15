import com.houarizegai.calculator.Calculator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.regex.Pattern;

class CalculatorTest {

    private Calculator calculator;

    @BeforeEach
    void setUp() { // Create object before compilation
        calculator = new Calculator();
    }

    @Test
    void PC_CC_01(){
        calculator.setVal("5");
        calculator.setGo(true);

        boolean b1 = Pattern.matches("([-]?\\d+[.]\\d*)|(\\d+)", calculator.inText.getText());
        boolean b2 = (calculator.getGo() == true);
        Assertions.assertTrue(b1 && b2);
    }

    @Test
    void PC_CC_02(){
        calculator.setVal("a");
        calculator.setGo(false);

        boolean b1 = Pattern.matches("([-]?\\d+[.]\\d*)|(\\d+)", calculator.inText.getText());
        boolean b2 = (calculator.getGo() == true);
        Assertions.assertFalse(b1 && b2);
    }

    @Test
    void CACC_01(){
        calculator.setVal("5");
        calculator.setGo(false);

        boolean b1 = Pattern.matches("([-]?\\d+[.]\\d*)|(\\d+)", calculator.inText.getText());
        boolean b2 = (calculator.getGo() == true);
        boolean b3 = Pattern.matches("[-]?[\\d]+[.][0]*", String.valueOf(calculator.val));
        Assertions.assertTrue(b1 || (b2 && b3));
    }

    @Test
    void CACC_02(){
        calculator.setVal("a");
        calculator.setGo(true);

        boolean b1 = Pattern.matches("([-]?\\d+[.]\\d*)|(\\d+)", calculator.inText.getText());
        boolean b2 = (calculator.getGo() == true);
        boolean b3 = Pattern.matches("[-]?[\\d]+[.][0]*", String.valueOf(calculator.val));
        Assertions.assertTrue(b1 || (b2 && b3));
    }

    @Test
    void CACC_03(){
        calculator.setVal("a");
        calculator.setGo(false);

        boolean b1 = Pattern.matches("([-]?\\d+[.]\\d*)|(\\d+)", calculator.inText.getText());
        boolean b2 = (calculator.getGo() == true);
        boolean b3 = Pattern.matches("[-]?[\\d]+[.][0]*", String.valueOf(calculator.val));
        Assertions.assertFalse(b1 || (b2 && b3));
    }

    @Test
    void Graph_01(){
        double first = 200;
        String second = "100";
        double answer = 200 + 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '+'));
    }
    @Test
    void Graph_02(){
        double first = 200;
        String second = "100";
        double answer = 200 - 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '-'));
    }
    @Test
    void Graph_03(){
        double first = 200;
        String second = "100";
        double answer = 200 * 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '*'));
    }
    @Test
    void Graph_04(){
        double first = 200;
        String second = "100";
        double answer = 200 / 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '/'));
    }
    @Test
    void Graph_05(){
        double first = 200;
        String second = "100";
        double answer = 200 % 100;
        Assertions.assertEquals(answer, calculator.calc(first, second, '%'));
    }
    @Test
    void Graph_06(){
        double first = 200;
        String second = "100";
        double answer = Math.pow(200,100);
        Assertions.assertEquals(answer, calculator.calc(first, second, '^'));
    }
    @Test
    void Graph_07(){
        double first = 3;
        String second = "10";
        double answer = Math.sqrt(10);
        Assertions.assertEquals(answer, calculator.calc(first, second, '√'));
    }
    @Test
    void Graph_08(){
        double first = 3;
        String second = "10";
        double answer = Math.log(10);
        Assertions.assertEquals(answer, calculator.calc(first, second, 'l'));
    }
    @Test
    void Graph_09(){
        calculator.setVal("5");
        calculator.btnC.doClick();
        Assertions.assertEquals("0", calculator.getResult());
    }

    @Test
    void Graph_10(){
        calculator.btnC.doClick();
        calculator.empty();
        Assertions.assertEquals("0", calculator.getResult());
    }
}
