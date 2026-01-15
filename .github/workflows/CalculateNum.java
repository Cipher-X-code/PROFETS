// write a java program to accept  infinite num from a user and calculate the count of positive  negative  your sentinel num
import java.util.Scanner;

public class CalculateNum {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        
        int num;
        int positiveCount = 0;
        int negativeCount = 0;

        System.out.println("Enter numbers (enter -1 to stop):");

        while (true) {
            System.out.print("Enter number: ");
            num = input.nextInt();

            if (num == -1) { 
                break;
            } else if (num > 0) {
                positiveCount++;
            } else if (num < 0) {
                negativeCount++;
            }
        }

        System.out.println("Count of positive numbers: " + positiveCount);
		
        System.out.println("Count of negative numbers: " + negativeCount);
		
        System.out.println("Sentinel number used: -1");
    }
}
