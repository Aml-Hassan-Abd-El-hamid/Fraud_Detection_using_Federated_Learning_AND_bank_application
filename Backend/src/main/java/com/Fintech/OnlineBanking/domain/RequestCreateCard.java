package com.Fintech.OnlineBanking.domain;

import java.time.LocalDate;

import java.time.ZonedDateTime;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

import org.apache.qpid.proton.amqp.messaging.Data;

import com.Fintech.OnlineBanking.user.User;
import com.fasterxml.jackson.annotation.JsonIgnore;

@Entity
@Table(name = "requestCreateCard")
public class RequestCreateCard {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long id;
	private long acountNumber;

	private LocalDate cardActivation;

	private String cardName;
	private String cardType;
	private long requestNumberCreateCard;

	private ZonedDateTime timeOfRequest = ZonedDateTime.now();

	@ManyToOne
	@JoinColumn(referencedColumnName = "user_info_national_id")
	private User user;


	public User getUser() {
		return user;
	}

	public String getCardType() {
		return cardType;
	}

	public void setCardType(String cardType) {
		this.cardType = cardType;
	}

	public void setUser(User user) {
		this.user = user;
	}

	public ZonedDateTime getTimeOfRequest() {
		return timeOfRequest;
	}

	public void setTimeOfRequest(ZonedDateTime timeOfRequest) {
		this.timeOfRequest = timeOfRequest;
	}

	public long getRequestNumberCreateCard() {
		return requestNumberCreateCard;
	}

	public void setRequestNumberCreateCard(long requestNumberCreateCard) {
		this.requestNumberCreateCard = requestNumberCreateCard;
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public long getAcountNumber() {
		return acountNumber;
	}

	public void setAcountNumber(long acountNumber) {
		this.acountNumber = acountNumber;
	}

	public LocalDate getCardActivation() {
		return cardActivation;
	}

	public void setCardActivation(LocalDate cardActivation) {
		this.cardActivation = cardActivation;
	}

	public String getCardName() {
		return cardName;
	}

	public void setCardName(String cardName) {
		this.cardName = cardName;
	}

}
