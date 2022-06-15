# Software_final_project Group 9

小組名單: 

- 310551124 官學勤 - Graph Coverage
- 310555020 鄭旭翔 - 整合、報告、CI
- 310555021 黃浩軒 - Logic Coverage
- 310581027 宋煜祥 - Test Case產出、問題回報、CI


測試項目: https://github.com/HouariZegai/Calculator
修改部分source code以方便測試


### Logic Coveage
測試PC、CC、CACC for PA(3、6、8)(2、4、7存在conflict)

主要測試
![image](https://user-images.githubusercontent.com/92283002/173741417-4681c74e-e501-494a-9600-e7ce1496c7ca.png)

此判斷式在程式中被大量重複使用

程式碼：
`CalculatorTest: PC_CC_01()、PC_CC_02()、CACC_01()、CACC_02()、CACC_03()`

### Graph Coverage
control flow graph
![image](https://user-images.githubusercontent.com/92283002/173732853-3c3acb1e-a022-414b-a4e9-8c12f6b016db.png)

主要測試運算部分(source code大部分參數為private難以測試)
![image](https://user-images.githubusercontent.com/92283002/173733418-c1ac2aa3-ce4f-460d-b644-f11094f73f41.png)

程式碼：
`CalculatorTest: Graph_01~Graph_10`

運行結果：

![image](https://user-images.githubusercontent.com/92283002/173752923-0f182ddf-48c1-4080-b5fb-d281e9aa00b2.png)

### CI
FAILED
原因：Github Action只支援head-less，無法用GUI測試Swing完成測試。
![image](https://user-images.githubusercontent.com/92283002/173734840-a9b6f8da-de36-4319-b362-7c586ca3b7f0.png)
