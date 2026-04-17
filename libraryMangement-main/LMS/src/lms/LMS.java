package lms;

import dbutil.DBConnection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Scanner;

public class LMS {
    public static void main(String[] args) throws SQLException {
        DBConnection database = new DBConnection();

        // Create a scanner object to read user input
        Scanner scanner = new Scanner(System.in);

        // Prompt the user to enter book details
        System.out.println("Enter book ID:");
        int bookID = scanner.nextInt();
        scanner.nextLine(); // Consume newline

        System.out.println("Enter book author:");
        String author = scanner.nextLine();

        System.out.println("Enter book name:");
        String name = scanner.nextLine();

        System.out.println("Enter book genre:");
        String genre = scanner.nextLine();

        System.out.println("Enter book price:");
        double price = scanner.nextDouble();
        scanner.nextLine(); // Consume newline

        System.out.println("Enter number of pages:");
        int pages = scanner.nextInt();
        scanner.nextLine(); // Consume newline

        System.out.println("Is the book available? (true/false):");
        boolean available = scanner.nextBoolean();
        scanner.nextLine(); // Consume newline

        // Create a new Book object
        Book book = new Book();
        book.setBookID(bookID);
        book.setAuthor(author);
        book.setName(name);
        book.setGenre(genre);
        book.setPrice(price);
        book.setPages(pages);
        book.setAvailable(available);

        // Add the book to the database
        boolean success = database.addBook(book);

        // Check if the insertion was successful
        if (success) {
            System.out.println("Book added successfully!");
        } else {
            System.out.println("Failed to add book.");
        }

        // Prompt the user to enter student details
        System.out.println("Enter student ID:");
        int id = scanner.nextInt();
        System.out.println("Enter student year:");
        int year = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        System.out.println("Enter student department:");
        String department = scanner.nextLine();
        System.out.println("Enter student full name:");
        String fullName = scanner.nextLine();
        System.out.println("Enter student book:");
        String studentBook = scanner.nextLine();

        // Create a new Student object
        Student student = new Student(id, year, department, fullName, studentBook);

        // Add the student to the database
        success = database.addStudent(student);

        // Check if the insertion was successful
        if (success) {
            System.out.println("Student added successfully!");
        } else {
            System.out.println("Failed to add student.");
        }

        // Display all students
        ArrayList<Student> students = database.displayAllStudents();
        for (Student s : students) {
            System.out.println(s);
        }

        // Display all books
        ArrayList<Book> books = database.displayAllBooks();
        for (Book b : books) {
            System.out.println(b);
        }

        // Close the database connection
        DBConnection.closeConnection(database.getConnection());

        // Add the method to add an account from user input
        addAccountFromUserInput();
    }

    // Method to add an account from user input
    public static void addAccountFromUserInput() {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter account name:");
        String name = scanner.nextLine();

        System.out.println("Enter account username:");
        String username = scanner.nextLine();

        System.out.println("Enter account password:");
        String password = scanner.nextLine();

        // Creating an Account object with user input
        Account account = new Account(username, password);

        // Create an instance of DBConnection to interact with the database
        DBConnection database = new DBConnection();

        // Call the addAccount method and store the result
        boolean success = database.addAccount(account);

        // Check the result and print appropriate message
        if (success) {
            System.out.println("Account added successfully!");
        } else {
            System.out.println("Failed to add account.");
        }
    }
}
