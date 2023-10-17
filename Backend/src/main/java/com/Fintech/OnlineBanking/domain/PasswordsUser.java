package com.Fintech.OnlineBanking.domain;

import java.io.Serializable;

import java.util.Date;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

import com.Fintech.OnlineBanking.user.User;

@Entity
@Table(name = "passwords")
public class PasswordsUser implements Serializable {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long id;
	private String password;
	@ManyToOne
	private User user;
	private Date passwordChangeTime;

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public Date getPasswordChangeTime() {
		return passwordChangeTime;
	}

	public void setPasswordChangeTime(Date passwordChangeTime) {
		this.passwordChangeTime = passwordChangeTime;
	}

	public User getUser() {
		return user;
	}

	public void setUser(User user) {
		this.user = user;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

}
