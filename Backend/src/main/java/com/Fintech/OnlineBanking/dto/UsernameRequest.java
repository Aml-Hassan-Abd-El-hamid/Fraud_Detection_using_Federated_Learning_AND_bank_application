package com.Fintech.OnlineBanking.dto;

import jakarta.validation.constraints.Digits;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public class UsernameRequest {
	// @Size(min=5, message = "Name should be atleast 5 characters")
	private String username;

	/* @Min(value=14, message = "nationalId should be 14 ")
	 @Max(value=14, message = "nationalId should be 14 ")*/
	 private Long nationalId ;

	public Long getNationalId() {
		return nationalId;
	}

	public void setNationalId(Long nationalId) {
		this.nationalId = nationalId;
	}

	public String getUsername() {
		return username;
	}

	public void setUsername(String username) {
		this.username = username;
	}

	

}
