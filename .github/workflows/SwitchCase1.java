import java.util.Scanner;
public class SwitchCase1 {
    public static void main(String [] args ){
        Scanner input = new Scanner(System.in);
        int num ;
        int i= 0;
        
        System.out.print("choose any  number you  want from (1-5):  ");
        num = input.nextInt();
        
          
        switch (num){
            case 1:

            System.out.print("Today is Sunday ");
            break;
            
            case 2 :
            System.out.print(" Today is monday ");
            break;
            
            case 3:
            System.out.print(" Today is tuesday");
            break;
            
            case 4:
            System.out.print(" Today is wednesday");
            break;
            
            case 5:
            System.out.print("Today is thursday ");
            break;
            
            case 6 :
            System.out.print("Today is friday ");
            break;
            
            case 7:
            System.out.print("Today is Saturday ");
            break;
           
            case 8:
       default:
       System.out.print("invalid  value ");
       
        
        }
       }
}

            
            
