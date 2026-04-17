package dbutil;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import lms.Book;
import lms.Student;
import lms.Account;
        
public class DBConnection {
    private static final String URL = "jdbc:sqlite:LibraryManagementDb.db";

    public static Connection getConnection() throws SQLException {
        try {
            Class.forName("org.sqlite.JDBC");
            return DriverManager.getConnection(URL);
        } catch (ClassNotFoundException ex) {
            throw new SQLException("Failed to establish database connection", ex);
        }
    }
    public boolean addAccount(Account account) {
    String sql = "INSERT INTO Account (UserName, Password) VALUES (?, ?)";
    try (Connection conn = getConnection();
         PreparedStatement pst = conn.prepareStatement(sql)) {
        pst.setString(1, account.getUserName());
        pst.setString(2, account.getPassword());
        pst.executeUpdate();
        return true;
    } catch (SQLException ex) {
        ex.printStackTrace();
        return false;
    }
}

     public boolean Login(String userName, String password) {
    String sql = "SELECT * FROM Account WHERE UserName = ? AND Password = ?";
    try (Connection conn = getConnection();
         PreparedStatement pst = conn.prepareStatement(sql)) {
        pst.setString(1, userName);
        pst.setString(2, password);
        try (ResultSet rs = pst.executeQuery()) {
            return rs.next(); // Returns true if there is at least one row, false otherwise
        }
    } catch (SQLException ex) {
        ex.printStackTrace();
        return false;
    }
}


    
    public boolean addBook(Book book) {
        String sql = "INSERT INTO Book (bookID,Name ,author, genre, available,Price,Pages ) VALUES (?, ?, ?, ?, ?,?,?)";
        try (Connection conn = getConnection();
             PreparedStatement pst = conn.prepareStatement(sql)) {
            pst.setInt(1, book.getBookID());
            pst.setString(2,book.getName());
            pst.setString(3, book.getAuthor());
            pst.setString(4, book.getGenre());
            pst.setBoolean(5, book.isAvailable());
            pst.setDouble(6, book.getPrice());
            pst.setInt(7, book.getPages());
            pst.executeUpdate();
            return true;
        } catch (SQLException ex) {
            ex.printStackTrace();
            return false;
        }
    }
    
    public ArrayList<Book> displayAllBooks() {
    ArrayList<Book> books = new ArrayList<>();
    String sql = "SELECT * FROM Book";
    try (Connection conn = getConnection();
         Statement smt = conn.createStatement();
         ResultSet rs = smt.executeQuery(sql)) {
        while (rs.next()) {
            int bookID = rs.getInt("bookID");
            String name = rs.getString("Name");
            String author = rs.getString("author");
            String genre = rs.getString("genre");
            boolean available = rs.getBoolean("available");
            double price = rs.getDouble("price");
            int pages = rs.getInt("Pages");
            Book book = new Book(bookID,name ,author, genre,available, price,pages );
            books.add(book);
        }
    } catch (SQLException ex) {
        ex.printStackTrace();
        return null;
    }
    return books;
}

    
    public boolean addStudent(Student student) {
        String sql = "INSERT INTO Student (id, year, department, fullName, book) VALUES (?, ?, ?, ?, ?)";
        try (Connection conn = getConnection();
             PreparedStatement pst = conn.prepareStatement(sql)) {
            pst.setInt(1, student.getId());
            pst.setInt(2, student.getYear());
            pst.setString(3, student.getDepartment());
            pst.setString(4, student.getFullName());
            pst.setString(5, student.getBook());
            pst.executeUpdate();
            return true;
        } catch (SQLException ex) {
            ex.printStackTrace();
            return false;
        }
    }
    
    public ArrayList<Student> displayAllStudents() {
        ArrayList<Student> students = new ArrayList<>();
        String sql = "SELECT * FROM Student"; 
        try (Connection conn = getConnection();
            Statement smt = conn.createStatement();
            ResultSet rs = smt.executeQuery(sql)) {
            while (rs.next()) {
                int year = rs.getInt(2);
                int id = rs.getInt(1);
                String department = rs.getString(3);
                String fullName = rs.getString(4);
                String book = rs.getString(5);
                students.add(new Student(year, id, department, fullName, book));
            }
        } catch (SQLException ex) {
            ex.printStackTrace();
        }
        return students;
    }

    public static void closeConnection(Connection connection) {
        if (connection != null) {
            try {
                connection.close();
            } catch (SQLException ex) {
                // Handle the exception or log it
                ex.printStackTrace();
            }
        }
    }
}
