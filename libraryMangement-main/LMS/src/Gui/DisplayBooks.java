package Gui;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;
import dbutil.DBConnection;
import lms.Book;

public class DisplayBooks extends JFrame {
    private DBConnection dbConnection;
    private JTable bookTable;

    public DisplayBooks() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Set title
        setTitle("Display Books");

        // Create JTable to display books
        bookTable = new JTable();
        JScrollPane scrollPane = new JScrollPane(bookTable);

        // Add scroll pane to the frame
        add(scrollPane);

        // Set frame properties
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    // Method to fetch and display books
    public void displayBooks() {
        // Get list of books from the database
        ArrayList<Book> books = dbConnection.displayAllBooks();

        // Create table model
        DefaultTableModel model = new DefaultTableModel();
        model.addColumn("Book ID");
        model.addColumn("Name");
        model.addColumn("Author");
        model.addColumn("Genre");
        model.addColumn("Available");
        model.addColumn("Price");
        model.addColumn("Pages");

        // Populate table model with book data
        for (Book book : books) {
            model.addRow(new Object[]{
                    book.getBookID(),
                    book.getName(),
                    book.getAuthor(),
                    book.getGenre(),
                    book.isAvailable(),
                    book.getPrice(),
                    book.getPages()
            });
        }

        // Set table model to the JTable
        bookTable.setModel(model);
    }

    public static void main(String[] args) {
        // Run the GUI on the event dispatch thread
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // Create an instance of DisplayBooks
                DisplayBooks displayBooks = new DisplayBooks();
                // Fetch and display books
                displayBooks.displayBooks();
                // Set the frame visible
                displayBooks.setVisible(true);
            }
        });
    }
}
